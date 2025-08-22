# src/core/loop.py
import logging
import time, os
import pandas as pd
import numpy as np
import random  # For random selection in diverse initial binders
import gc  # <--- NEW: Import Garbage Collector
import torch  # <--- NEW: Import torch to clear cache
from Bio import SeqIO
from pathlib import Path
from src.data_access.manager import DataManager
from src.models.generator import BinderGenerator
from src.models.structural_predictor import AlphaFold3Predictor
from src.models.sequence_encoder import SequenceEncoder
from src.models.surrogate_model import SurrogateModel
from src.core.mutagenesis import Mutagenesis
from src.core.acquisition import AcquisitionFunction
from src.models.redesigner import ProteinMPNNRedesigner
from src.filters.rules import BinderFilters

logger = logging.getLogger(__name__)


class ActiveLearningOrchestrator:
    """
    Orchestrates the active learning pipeline, managing the flow between
    binder generation, structural prediction, and data management.
    """

    def __init__(self, config):
        self.config = config
        self._validate_target_sequences()
        self.data_manager = DataManager(config)
        self.binder_generator = BinderGenerator(config)
        self.af3_predictor = AlphaFold3Predictor(config)
        self.esm_encoder = SequenceEncoder(config)
        self.surrogate_model = SurrogateModel(config)  # For ultimate property prediction, if added later
        self.mutagenesis = Mutagenesis(config)
        self.mpnn_redesigner = ProteinMPNNRedesigner(config)
        self.binder_filters = BinderFilters(config)
        self.acquisition_function = AcquisitionFunction(config)
        self.all_seen_sequences = set()  # Stores binder_sequence strings

        # # Load models once at initialization
        # try:
        #     self.binder_generator.load_model()
        #     if self.config.models.esm_encoder.enabled:
        #         self.esm_encoder.load_model()
        # except Exception as e:
        #     logger.critical(f"Failed to load a required model: {e}. Exiting.")
        #     raise

    def _validate_target_sequences(self):
        """
        Validates that the sequence in the target FASTA file matches the sequence(s)
        in the target CIF file (specifically chain A, if applicable for your CIF).
        Halts execution if they do not match.
        """
        logger.info("Validating target FASTA sequence against target CIF sequence...")

        target_fasta_path = self.config.paths.get('target_fasta')
        target_cif_path = self.config.paths.get('target_cif_path')

        if not target_fasta_path or not os.path.exists(target_fasta_path):
            logger.critical(f"Target FASTA file not found at configured path: {target_fasta_path}. Halting.")
            raise FileNotFoundError(f"Target FASTA file not found: {target_fasta_path}")
        if not target_cif_path or not os.path.exists(target_cif_path):
            logger.critical(f"Target CIF file not found at configured path: {target_cif_path}. Halting.")
            raise FileNotFoundError(f"Target CIF file not found: {target_cif_path}")

        # 1. Read sequence from FASTA (assuming single sequence or first sequence)
        fasta_seq_record = None
        try:
            fasta_seq_record = next(SeqIO.parse(target_fasta_path, "fasta"))
            fasta_sequence = str(fasta_seq_record.seq).upper().replace("*", "")  # Remove stop codons if any
            logger.debug(
                f"Sequence from FASTA ({Path(target_fasta_path).name}): {fasta_sequence[:30]}... (len: {len(fasta_sequence)})")
        except Exception as e:
            logger.critical(f"Error reading target FASTA file {target_fasta_path}: {e}. Halting.")
            raise

        cif_sequences = {}
        try:
            from Bio.PDB import MMCIFParser, Polypeptide
            from Bio.SeqUtils import seq1  # To convert 3-letter to 1-letter AA codes

            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("target_cif", target_cif_path)

            for model in structure:
                for chain in model:
                    chain_id = chain.id
                    seq = ""
                    for residue in chain:
                        # Skip hetero residues and waters
                        if residue.id[0] != " ":
                            continue
                        # Get the 3-letter residue name
                        resname = residue.resname
                        # Convert to 1-letter code
                        try:
                            one_letter = seq1(resname)
                        except KeyError:
                            one_letter = 'X'  # Unknown residue
                        seq += one_letter
                    cif_sequences[chain_id] = seq

            if not cif_sequences:
                logger.critical(f"Could not extract any sequences from target CIF file {target_cif_path}. Halting.")
                raise ValueError(f"No sequences found in CIF: {target_cif_path}")

        except ImportError:
            logger.critical(f"Bio.PDB.MMCIFParser not found. Cannot parse CIF. Halting.")
            raise
        except Exception as e:
            logger.critical(f"Error reading or parsing target CIF file {target_cif_path}: {e}. Halting.")
            raise

        # 3. Compare (assuming target chain in CIF is 'A' or the first one found if only one)
        # Adjust this logic based on how your target is represented in the CIF.
        target_chain_in_cif = 'A'  # Or determine dynamically if needed

        if target_chain_in_cif not in cif_sequences:
            if len(cif_sequences) == 1:  # If only one chain, assume it's the target
                target_chain_in_cif = list(cif_sequences.keys())[0]
                logger.warning(
                    f"Target chain 'A' not found in CIF, using the only available chain: '{target_chain_in_cif}'.")
            else:
                logger.critical(
                    f"Target chain '{target_chain_in_cif}' not found in CIF file {target_cif_path}. Available chains: {list(cif_sequences.keys())}. Halting.")
                raise ValueError(f"Target chain '{target_chain_in_cif}' not found in CIF: {target_cif_path}")

        cif_sequence_chain_a = cif_sequences[target_chain_in_cif].upper().replace("*", "")
        logger.debug(
            f"Sequence from CIF (Chain {target_chain_in_cif}, {Path(target_cif_path).name}): {cif_sequence_chain_a[:30]}... (len: {len(cif_sequence_chain_a)})")

        if fasta_sequence != cif_sequence_chain_a:
            logger.critical("CRITICAL MISMATCH: Target sequence in FASTA does not match sequence in CIF (Chain A).")
            logger.critical(f"FASTA ({len(fasta_sequence)}aa): {fasta_sequence}")
            logger.critical(f"CIF   ({len(cif_sequence_chain_a)}aa): {cif_sequence_chain_a}")
            raise ValueError("Target FASTA and CIF sequences do not match. Halting execution.")
        else:
            logger.info("Target FASTA and CIF sequences match successfully.")

    def _load_and_release_model(self, model_instance, load: bool = True):
        """Helper to load or release a model from GPU memory."""
        if load:
            model_instance.load_model()
        else:
            model_name = model_instance.__class__.__name__
            logger.info(f"Releasing model: {model_name}")
            del model_instance.model
            del model_instance.tokenizer
            model_instance.model = None
            model_instance.tokenizer = None
            gc.collect()  # Trigger garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear PyTorch's CUDA cache
            logger.info(f"Model {model_name} released from memory.")

    def _process_binders_batch(self, binders_data_list, iteration_num, keywords='no_pmpnn'):
        """
        Helper function to process a batch of binders through AF3.
        Filters are NOT applied here, as per requirements.
        """
        if not binders_data_list:
            logger.warning(f"No binders to process in Iteration {iteration_num}. Returning empty list.")
            return []

        logger.info(f"Processing {len(binders_data_list)} binders with AF3 for Iteration {iteration_num}.")

        binders_with_af3_results = self.af3_predictor.predict_complexes(
            binders_data_list=binders_data_list,
            iteration_num=iteration_num
        )
        logger.info(f"Completed AlphaFold3 predictions for {len(binders_with_af3_results)} binders.")

        alpha = self.config.active_learning.weight_for_iptm
        ptm_w = self.config.active_learning.weight_for_ptm
        plddt_w = self.config.active_learning.weight_for_plddt
        pae_w = self.config.active_learning.weight_for_pae
        ipsae_w = self.config.active_learning.weight_for_ipsae
        sc_w = self.config.active_learning.weight_for_sc
        hs_w = self.config.active_learning.weight_for_hs
        unsH_w = self.config.active_learning.weight_for_unsH
        for binder_entry in binders_with_af3_results:
            binder_entry["iteration"] = iteration_num
            if 'passed_filters' not in binder_entry:
                binder_entry['passed_filters'] = None
                binder_entry['filter_reason'] = "Not filtered in AL loop"
            if (binder_entry.get('af3_status') == 'success' and
                    binder_entry.get('iptm') is not None and
                    binder_entry.get('ptm') is not None):
                # binder_entry['combined_score'] = (alpha * binder_entry['iptm']) + ((1 - alpha) * binder_entry['ptm'])
                binder_entry['combined_score'] = ((alpha * binder_entry['iptm']) +
                                                  ptm_w * binder_entry['ptm'] +
                                                  plddt_w * binder_entry['binder_plddt'] +
                                                  pae_w * binder_entry['pae'] / 37 +
                                                  ipsae_w * binder_entry['ipsae_min'] +
                                                  sc_w * binder_entry['shape_complementarity'] +
                                                  hs_w * binder_entry['surface_hydrophobicity'] +
                                                  unsH_w * binder_entry['interface_delta_unsat_hbonds'] / 30
                                                  )

            else:
                binder_entry['combined_score'] = None  # Set to None if metrics are missing

        self.data_manager.save_iteration_results(iteration_num, binders_with_af3_results, keywords=keywords)

        logger.info(
            f"Iteration {iteration_num} AF3 processing complete. Total processed: {len(binders_with_af3_results)}.")
        return binders_with_af3_results

    def _generate_and_store_embeddings(self, binders_data: list, iteration_num: int):
        """
        Helper to generate ESM embeddings for a list of binders and update their entries.
        Embeddings are saved as .npy files, and paths are stored in binder_entry.
        """
        if not self.config.models.esm_encoder.enabled:
            logger.info("ESM encoder disabled. Skipping embedding generation.")
            return

        logger.info("Loading ESM Encoder for embedding generation...")
        self._load_and_release_model(self.esm_encoder, load=True)

        logger.info(f"Generating embeddings for {len(binders_data)} binders...")
        sequences_for_embedding = []
        # Create a mapping from id to the actual binder_entry object for efficient updates later.
        binder_id_to_entry_map = {b['id']: b for b in binders_data}

        for b_entry in binders_data:
            if b_entry.get('binder_sequence') and isinstance(b_entry['binder_sequence'], str) and b_entry[
                'binder_sequence'].strip():
                af3_status = b_entry.get('af3_status')
                if af3_status is None or af3_status not in ["failed_input_prep", "failed_subprocess",
                                                            "failed_executable", "failed_general",
                                                            "failed_batch_subprocess", "skipped"]:
                    sequences_for_embedding.append({'id': b_entry['id'], 'sequence': b_entry['binder_sequence']})
                else:
                    logger.debug(
                        f"Skipping embedding generation for binder {b_entry.get('id', 'N/A')} due to AF3 status: {af3_status}")
                    b_entry['embedding_path'] = None
            else:
                logger.warning(
                    f"Skipping embedding generation for binder {b_entry.get('id', 'N/A')} due to missing or invalid sequence.")
                b_entry['embedding_path'] = None

        # Call ESM encoder
        # Ensure iteration_num is passed to SequenceEncoder.get_embeddings
        embeddings_results = self.esm_encoder.get_embeddings(sequences_for_embedding, iteration_num)

        # Map embedding paths back to the original binder data list
        for emb_res in embeddings_results:
            binder_id = emb_res['id']
            if binder_id in binder_id_to_entry_map:
                binder_entry = binder_id_to_entry_map[binder_id]
                binder_entry['embedding_path'] = emb_res['embedding_path']
            else:
                logger.warning(f"Embedding generated for unknown binder ID: {binder_id}")

        # Ensure all original binders_data entries have an 'embedding_path' key
        for binder_entry in binders_data:
            if 'embedding_path' not in binder_entry:
                binder_entry['embedding_path'] = None

        logger.info("Embeddings generated and paths added to binder data.")

        self._load_and_release_model(self.esm_encoder, load=False)  # Release after use
        logger.info("ESM Encoder released.")

    def _select_top_N_diverse_initial_binders(self, current_iteration: int = 0):
        """
        Generates a large initial pool of binders, processes them through AF3,
        generates embeddings, and then selects a diverse, top-performing subset
        to kickstart the Active Learning loop.
        """
        logger.info(f"Starting initial diverse binder selection (Iteration {current_iteration}).")

        targets = self.data_manager.load_initial_targets()
        if not targets:
            logger.error("No targets found to start generation. Exiting.")
            return []
        logger.info(f"Loaded {len(targets)} targets for initial generation.")

        # 1. Generate a large pool of initial binders (e.g., 100 sequences)
        logger.info("Loading Binder Generator for initial generation...")
        self._load_and_release_model(self.binder_generator, load=True)

        num_initial_gen_sequences = self.config.models.seq2seq.generation_params.num_return_sequences
        logger.info(f"Generating {num_initial_gen_sequences} initial binders from Seq2Seq model.")
        generated_binders_data = self.binder_generator.generate_binders(
            targets_to_generate=targets,
            num_return_sequences=num_initial_gen_sequences  # Use the large number here
        )
        # Update all_seen_sequences with these initial generated sequences
        self.all_seen_sequences.update(b['binder_sequence'] for b in generated_binders_data if 'binder_sequence' in b)
        logger.info(f"Generated {len(generated_binders_data)} total unique initial binder candidates.")

        # 2. Process ALL generated binders through AF3 (Iteration 0)
        # This will save iteration_0_results.csv and update al_state.csv with AF3 results (no embeddings yet)
        initial_evaluated_all_binders = self._process_binders_batch(generated_binders_data, current_iteration)

        # 3. Generate Embeddings for ALL initial evaluated binders
        # This modifies initial_evaluated_all_binders (in-memory) with embedding_path
        self._generate_and_store_embeddings(initial_evaluated_all_binders, current_iteration)

        # 5. Select the diverse top N initial binders for the AL loop
        # Filter for successfully AF3-evaluated binders with embeddings, from this large pool
        eligible_initial_parents = [
            b for b in initial_evaluated_all_binders
            if b.get('af3_status') == 'success' and
               b.get(self.acquisition_function.surrogate_model_config.target_metric) is not None and
               b.get('embedding_path') is not None and  # Check for loaded embedding
               b.get('is_in_defined_binding_region')
        ]

        if not eligible_initial_parents:
            logger.warning(
                "No binders successfully evaluated with embeddings in initial batch. Cannot select diverse parents.")
            return []

        num_diverse_groups = self.config.active_learning.num_initial_diversity_groups
        # Pass the full list of eligible initial binders to the diversity selector
        # This function will handle loading embeddings as needed internally if passed as paths
        selected_initial_parents = self.acquisition_function.select_top_N_diverse_from_pool(
            eligible_initial_parents,
            num_to_select=self.config.active_learning.num_parent_binders_to_optimize,
            num_diversity_groups=num_diverse_groups,
            metric_for_selection=self.acquisition_function.surrogate_model_config.target_metric,
            # Use the defined target metric
            data_manager=self.data_manager  # Pass data manager for embedding loading
        )

        logger.info(f"Selected {len(selected_initial_parents)} diverse top binders on "
                    f"{self.acquisition_function.surrogate_model_config.target_metric} to seed the AL loop.")

        # 4. Save selected_initial_parents to AL state
        # This ensures all generated and embedded initial binders are in al_state.csv
        self.data_manager.update_al_state(selected_initial_parents)
        logger.info(
            f"Updated overall AL state with all {len(selected_initial_parents)} Iteration {current_iteration} binders (AF3 and embeddings).")

        self._load_and_release_model(self.binder_generator, load=False)  # Release after use
        logger.info("Binder Generator released.")
        return selected_initial_parents  # These are the actual parents for the first iteration

    def _identify_solved_lineages(self, evaluated_data: list):
        """
        NEW HELPER: Identifies original_parent_id lineages that are considered "solved".
        """
        solved_lineages = set()
        if not evaluated_data:
            return solved_lineages

        df = pd.DataFrame(evaluated_data)
        # Ensure target_metric column exists (e.g., 'combined_score')
        target_metric_name = self.surrogate_model.target_metric  # From surrogate model config
        if target_metric_name not in df.columns:
            logger.warning(f"Target metric '{target_metric_name}' not found in data for identifying solved lineages.")
            return solved_lineages

        # Filter for successful AF3 predictions with valid target metric
        df_successful = df[
            (df['af3_status'] == 'success') &
            (df[target_metric_name].notna())
            ].copy()

        if df_successful.empty:
            return solved_lineages

        threshold_value = self.config.active_learning.solved_lineage_threshold_metric_value
        min_count = self.config.active_learning.solved_lineage_min_binders_count

        for original_id, group_df in df_successful.groupby('original_parent_id'):
            num_high_scoring = len(group_df[group_df[target_metric_name] >= threshold_value])
            if num_high_scoring >= min_count:
                solved_lineages.add(original_id)
                logger.info(
                    f"Lineage '{original_id}' identified as SOLVED ({num_high_scoring} binders >= {threshold_value}).")
        return solved_lineages

    def run_initial_generation_and_evaluation(self):
        """
        Wrapper to call the diverse initial binder selection.
        This serves as 'iteration 0' of the active learning process.
        """
        current_iteration = 0
        logger.info(f"Starting BindSculpt Initial Generation and AF3 Evaluation Phase (Iteration {current_iteration}).")

        # This now calls the dedicated helper for diverse initial selection
        initial_evaluated_binders = self._select_top_N_diverse_initial_binders(current_iteration)

        if not initial_evaluated_binders:
            logger.warning("No binders selected from initial diverse pool. Cannot proceed with active learning.")
            return []  # Return empty if selection failed

        logger.info(
            f"Initial generation and AF3 evaluation complete. Selected {len(initial_evaluated_binders)} diverse binders for AL.")

        return initial_evaluated_binders

    def run_full_active_learning_loop(self, start_iteration: int = 0):
        """
        Orchestrates the full multi-iteration active learning loop,
        using a surrogate model for prediction and AF3 for true evaluation.
        ProteinMPNN and final filtering are external to this loop.
        """
        num_al_iterations = self.config.active_learning.num_iterations
        logger.info(f"Starting full Active Learning loop for {num_al_iterations} iterations.")

        # --- Iteration 0: Initial generation and evaluation ---
        # This populates al_state.csv with the initially selected, diverse, high-IPTm binders
        # These are the first parents for the mutagenesis.
        initial_evaluated_parents_for_al = ''
        if start_iteration == 0:
            # If starting from scratch, run Iteration 0 to generate initial data.
            initial_evaluated_parents_for_al = self.run_initial_generation_and_evaluation()
            if not initial_evaluated_parents_for_al:
                logger.warning("No initial parents selected for AL loop. Cannot proceed.")
                return
            # Initialize the set of seen sequences from Iteration 0
            self.all_seen_sequences.update(b['binder_sequence'] for b in initial_evaluated_parents_for_al)
        else:
            # If resuming, load ALL previously generated/evaluated sequences to populate `all_seen_sequences`.
            logger.info(f"Resuming run. Populating 'all_seen_sequences' from al_state.csv...")
            all_binders_df = self.data_manager.get_al_state()
            if not all_binders_df.empty:
                self.all_seen_sequences.update(all_binders_df['binder_sequence'].dropna().unique())
            logger.info(f"Resumed with {len(self.all_seen_sequences)} unique sequences seen so far.")

            # The loop now starts from the specified iteration (or 1 if starting fresh)
        start_loop_at = max(1, start_iteration)

        candidates_for_final_redesign = []
        final_iteration_num = 0
        for i in range(start_loop_at, num_al_iterations + 1):
            logger.info(f"\n--- Running Active Learning Iteration {i}/{num_al_iterations} ---")

            mpnn_redesigned_and_evaluated = []
            if (self.mpnn_redesigner.enabled and
                    ((i > 0 and i % self.config.active_learning.proteinmpnn_redesign_frequency == 0) or i == 1)):

                logger.info(f"--- Iteration {i}: Performing periodic ProteinMPNN redesign ---")

                # The candidates for MPNN redesign should be the *current best actual performers*
                # from each of the *original parent lineages* that are still active (not "solved").
                # We use `training_data_for_proxy` as it contains all AF3-evaluated data with scores.

                # First, identify unsolved lineages to focus MPNN on them.
                all_evaluated_binders_df = self.data_manager.get_al_state()
                all_evaluated_binders_data = self.data_manager.load_embeddings_for_binders(
                    all_evaluated_binders_df.to_dict(orient='records')
                )

                # Filter training data for successful AF3 predictions, valid embeddings, and relevant metrics
                target_metric_name = self.surrogate_model.target_metric
                training_data_for_proxy = [
                    b for b in all_evaluated_binders_data
                    if b.get('af3_status') == 'success' and
                       b.get(target_metric_name) is not None and
                       b.get('embedding_path') is not None and
                       b.get('is_in_defined_binding_region')
                ]
                solved_lineage_ids_for_mpnn = self._identify_solved_lineages(training_data_for_proxy)
                unsolved_training_data_for_mpnn = [
                    b for b in training_data_for_proxy  # Use the full list of evaluated data
                    if b.get('original_parent_id') not in solved_lineage_ids_for_mpnn and
                       b.get('af3_status') == 'success'  # Ensure they have a PDB
                ]

                if not unsolved_training_data_for_mpnn:
                    logger.info("No suitable (unsolved) binders found for ProteinMPNN redesign in this iteration.")
                else:
                    # Select the single best descendant from each UNSOLVED original parent lineage
                    # This uses the actual AF3 scores.
                    best_from_each_unsolved_lineage_for_mpnn = self.acquisition_function.select_candidates(
                        binders_data_list=unsolved_training_data_for_mpnn,
                        batch_size=1,  # Select only 1 best per group
                        strategy=self.acquisition_function.surrogate_model_config.target_metric,
                        # Use actual scores (e.g., combined_score)
                        group_by_key='original_parent_id'
                    )

                    # Further filter these best-of-lineage candidates by proteinmpnn_redesign_threshold
                    # and limit to proteinmpnn_num_candidates_to_redesign
                    binders_for_mpnn_redesign = []
                    if best_from_each_unsolved_lineage_for_mpnn:
                        metric_for_mpnn_selection = self.surrogate_model.target_metric
                        threshold = 0 if i == 1 else self.config.active_learning.proteinmpnn_redesign_threshold

                        eligible_for_mpnn = [
                            b for b in best_from_each_unsolved_lineage_for_mpnn
                            if b.get(metric_for_mpnn_selection, -1) >= threshold and b.get('af3_pdb_path')
                        ]

                        # Sort and take top N if more than configured
                        num_to_redesign = self.config.active_learning.proteinmpnn_num_candidates_to_redesign
                        binders_for_mpnn_redesign = sorted(
                            eligible_for_mpnn,
                            key=lambda x: x.get(metric_for_mpnn_selection, -float('inf')),
                            reverse=True
                        )[:num_to_redesign]

                    if binders_for_mpnn_redesign:
                        logger.info(
                            f"Selected {len(binders_for_mpnn_redesign)} top-performing binders (best of each unsolved lineage, above threshold) for ProteinMPNN redesign.")

                        redesigned_by_mpnn_raw = self.mpnn_redesigner.redesign_binders_batch(
                            binders_to_redesign_data=binders_for_mpnn_redesign,
                            iteration_num=i
                        )

                        if redesigned_by_mpnn_raw:
                            # Filter for unique sequences and evaluate with AF3
                            # ... (unique MPNN sequence filtering as before) ...
                            # ... (direct AF3 evaluation via _process_binders_batch as before) ...
                            # ... (embedding and updating AL state for these as before) ...
                            unique_mpnn_sequences_for_af3 = []
                            num_mpnn_duplicates_removed = 0
                            for mpnn_entry in redesigned_by_mpnn_raw:
                                if mpnn_entry['binder_sequence'] not in self.all_seen_sequences:
                                    unique_mpnn_sequences_for_af3.append(mpnn_entry)
                                    self.all_seen_sequences.add(mpnn_entry['binder_sequence'])
                                else:
                                    num_mpnn_duplicates_removed += 1
                            logger.info(
                                f"Removed {num_mpnn_duplicates_removed} duplicate MPNN sequences. Remaining for AF3: {len(unique_mpnn_sequences_for_af3)}")

                            if unique_mpnn_sequences_for_af3:
                                logger.info(
                                    f"Evaluating {len(unique_mpnn_sequences_for_af3)} ProteinMPNN redesigned binders with AF3.")
                                mpnn_redesigned_and_evaluated = self._process_binders_batch(
                                    unique_mpnn_sequences_for_af3, iteration_num=i, keywords='pmpnn'
                                )
                                self._generate_and_store_embeddings(mpnn_redesigned_and_evaluated, i)
                                self.data_manager.update_al_state(mpnn_redesigned_and_evaluated)
                                logger.info(
                                    f"Added {len(mpnn_redesigned_and_evaluated)} AF3-evaluated ProteinMPNN binders to AL state.")
                    else:
                        logger.info(
                            "No binders met criteria for ProteinMPNN redesign in this iteration after filtering.")
            # --- END Periodic ProteinMPNN Redesign ---

            # 1. Prepare Training Data for AF3 Score Proxy Model
            all_evaluated_binders_df = self.data_manager.get_al_state()
            all_evaluated_binders_data = self.data_manager.load_embeddings_for_binders(
                all_evaluated_binders_df.to_dict(orient='records')
            )

            # Filter training data for successful AF3 predictions, valid embeddings, and relevant metrics
            target_metric_name = self.surrogate_model.target_metric
            training_data_for_proxy = [
                b for b in all_evaluated_binders_data
                if b.get('af3_status') == 'success' and
                   b.get(target_metric_name) is not None and
                   b.get('embedding_path') is not None and
                   b.get('is_in_defined_binding_region')
            ]
            logger.info(f"Training AF3 Score Proxy with {len(training_data_for_proxy)} truly evaluated data points.")

            # 2. Train AF3 Score Proxy Model
            self.surrogate_model.train(training_data_for_proxy,
                                       iteration_num=i)  # Note: this is actually AF3 score proxy model

            # 3. Select Parent Binders for Mutagenesis
            solved_lineage_ids = self._identify_solved_lineages(training_data_for_proxy)
            unsolved_training_data_for_parents = [
                b for b in training_data_for_proxy
                if b.get('original_parent_id') not in solved_lineage_ids
            ]
            logger.info(
                f"Considering {len(unsolved_training_data_for_parents)} binders from unsolved lineages for parent selection.")

            parent_binders_for_mutagenesis = self.acquisition_function.select_candidates(
                unsolved_training_data_for_parents,
                batch_size=self.config.active_learning.num_binders_to_select_in_each_parent_for_mutagenesis,
                group_by_key='original_parent_id',
                strategy=self.acquisition_function.surrogate_model_config.target_metric
                # Strategy to pick the best parents based on actual AF3 scores
            )
            logger.info(
                f"Iteration {i}: Using diverse parents (total {len(parent_binders_for_mutagenesis)}) for mutagenesis.")

            if not parent_binders_for_mutagenesis:
                logger.warning("No suitable parent binders found for mutagenesis. Stopping AL loop.")
                break
            logger.info(f"Selected {len(parent_binders_for_mutagenesis)} parent binders for mutagenesis.")

            # 4. Generate NEW Mutant Pool for all selected parents
            newly_generated_mutant_pool = []

            current_mutation_strategy = self.config.mutagenesis.mutation_strategy_per_iteration
            if current_mutation_strategy == "alternate":
                iteration_mutation_type = "single_site" if i % 2 != 0 else "multiple_site"
            else:
                iteration_mutation_type = current_mutation_strategy

            logger.info(f"Iteration {i}: Using mutation strategy: {iteration_mutation_type}.")
            # Define number of mutations for multiple_site based on config

            for parent_binder in parent_binders_for_mutagenesis:
                parent_original_id = parent_binder.get('original_parent_id', parent_binder['id'])

                if iteration_mutation_type == "single_site":
                    mutants = self.mutagenesis.generate_single_mutants(parent_sequence=parent_binder['binder_sequence'],
                                                                       parent_id=parent_binder['id'],
                                                                       original_parent_id=parent_original_id,
                                                                       iteration_num=i
                                                                       )
                elif iteration_mutation_type == "multiple_site":
                    mutants = self.mutagenesis.generate_random_multiple_mutants(
                        parent_sequence=parent_binder['binder_sequence'],
                        parent_id=parent_binder['id'],
                        original_parent_id=parent_original_id,
                        iteration_num=i
                    )
                else:
                    logger.error(
                        f"Unknown mutation strategy '{iteration_mutation_type}'. Skipping mutagenesis for parent {parent_binder['id']}.")
                    mutants = []
                newly_generated_mutant_pool.extend(mutants)
            logger.info(f"Generated a pool of {len(newly_generated_mutant_pool)} NEW mutants from parent binders.")

            unique_newly_generated_mutant_pool = []
            num_duplicates_removed_this_iter = 0
            for mutant_entry in newly_generated_mutant_pool:
                if mutant_entry['binder_sequence'] not in self.all_seen_sequences:
                    unique_newly_generated_mutant_pool.append(mutant_entry)
                    self.all_seen_sequences.add(mutant_entry['binder_sequence'])
                else:
                    num_duplicates_removed_this_iter += 1

            logger.info(
                f"Removed {num_duplicates_removed_this_iter} duplicate sequences from NEWLY generated mutant pool. Remaining: {len(unique_newly_generated_mutant_pool)}")
            newly_generated_mutant_pool = unique_newly_generated_mutant_pool

            if not newly_generated_mutant_pool:
                logger.warning(
                    "Newly generated mutant pool became empty after removing duplicates. Cannot proceed with prediction for this iteration.")
                pass

            # 5. Generate Embeddings for the NEWLY Generated Mutant Pool
            self._generate_and_store_embeddings(unique_newly_generated_mutant_pool, i)
            self.data_manager.update_al_state(unique_newly_generated_mutant_pool)
            logger.info(
                f"Updated overall AL state with {len(unique_newly_generated_mutant_pool)} newly generated and embedded mutants from Iteration {i}.")

            # Build mutant_pool_for_prediction from ALL UN-EVALUATED mutants
            full_al_state_current_df = self.data_manager.get_al_state()
            full_al_state_current_data = self.data_manager.load_embeddings_for_binders(
                full_al_state_current_df.to_dict(orient='records')
            )

            already_evaluated_ids = set(b['id'] for b in training_data_for_proxy)
            already_evaluated_ids.update(b['id'] for b in mpnn_redesigned_and_evaluated)

            mutant_pool_for_prediction = [
                b for b in full_al_state_current_data
                if b.get('embedding_path') is not None and
                   b.get('embedding') is not None and
                   b.get('id') not in already_evaluated_ids and
                   b.get('af3_status') != 'success'  # Explicitly ensure it wasn't AF3-evaluated
            ]

            if not mutant_pool_for_prediction:
                logger.warning("No new, valid, and unevaluated mutants in pool for prediction. Stopping AL loop.")
                break
            logger.info(
                f"Ready to predict scores for {len(mutant_pool_for_prediction)} new mutants (excluding already evaluated).")

            # 6. Predict AF3 Metrics for the Mutant Pool using Surrogate Model (AF3 Score Proxy)
            # predicted_mutants = self.surrogate_model.predict(mutant_pool_for_prediction)  # This is actually AF3 proxy
            predicted_mutants = self.surrogate_model.predict_with_uncertainty(
                mutant_pool_for_prediction)  # with_uncertainty

            # 7.1. Acquire Candidates for TRUE AF3 Evaluation (DIFFERENT QUOTAS FOR SOLVED VS UNSOLVED)
            candidates_for_af3_evaluation = []

            # Separate predicted_mutants into solved and unsolved based on their original_parent_id
            predicted_mutants_solved_lineages = [
                m for m in predicted_mutants if m.get('original_parent_id') in solved_lineage_ids
            ]
            predicted_mutants_unsolved_lineages = [
                m for m in predicted_mutants if m.get('original_parent_id') not in solved_lineage_ids
            ]

            # Select from UNSOLVED lineages using regular quota
            if predicted_mutants_unsolved_lineages:
                selected_from_unsolved = self.acquisition_function.select_candidates(
                    binders_data_list=predicted_mutants_unsolved_lineages,
                    strategy=self.config.active_learning.selection_strategy_for_evaluation,
                    num_top=self.config.active_learning.num_top_candidates_per_group,
                    num_bottom=self.config.active_learning.num_bottom_candidates_per_group,
                    num_uncertain=self.config.active_learning.num_uncertain_candidates_per_group,
                    group_by_key='original_parent_id'
                )
                candidates_for_af3_evaluation.extend(selected_from_unsolved)
                logger.info(f"Selected {len(selected_from_unsolved)} candidates from UNSOLVED lineages for AF3.")

            # Select from SOLVED lineages using REDUCED quota (e.g., 1 top, 1 bottom, 1 uncertain)
            if predicted_mutants_solved_lineages:
                selected_from_solved = self.acquisition_function.select_candidates(
                    binders_data_list=predicted_mutants_solved_lineages,
                    strategy=self.config.active_learning.selection_strategy_for_evaluation,  # Same strategy
                    num_top=self.config.active_learning.solved_lineage_num_top_candidates,
                    num_bottom=self.config.active_learning.solved_lineage_num_bottom_candidates,
                    num_uncertain=self.config.active_learning.solved_lineage_num_uncertain_candidates,
                    group_by_key='original_parent_id'
                )
                candidates_for_af3_evaluation.extend(selected_from_solved)
                logger.info(
                    f"Selected {len(selected_from_solved)} candidates from SOLVED lineages (reduced quota) for AF3.")

            if not candidates_for_af3_evaluation:
                logger.warning(f"No candidates selected for true AF3 evaluation in Iteration {i}. Stopping AL loop.")
                break

            logger.info(f"Total {len(candidates_for_af3_evaluation)} candidates selected for true AF3 evaluation.")

            # 8. Augment selected candidates with target info
            parent_lookup = {b['id']: {'target_id': b['target_id'], 'target_sequence': b['target_sequence']}
                             for b in training_data_for_proxy}

            augmented_candidates_for_af3 = []
            for candidate_mutant in candidates_for_af3_evaluation:
                parent_id = candidate_mutant.get('parent_id')
                # For mutants, the target_id and target_sequence come from their immediate parent
                # We need to ensure the parent is in the training_data_for_proxy or original parents
                if parent_id and parent_id in parent_lookup:
                    candidate_mutant['target_id'] = parent_lookup[parent_id]['target_id']
                    candidate_mutant['target_sequence'] = parent_lookup[parent_id]['target_sequence']
                    augmented_candidates_for_af3.append(candidate_mutant)
                else:
                    # If parent_id is missing or parent not found, it might be an initial Seq2Seq binder
                    # or an issue in data consistency. Log critical and skip for now.
                    if 'target_id' in candidate_mutant and 'target_sequence' in candidate_mutant:
                        augmented_candidates_for_af3.append(
                            candidate_mutant)  # Already has target info (e.g., initial binder)
                    else:
                        logger.error(
                            f"Parent '{parent_id}' not found for mutant '{candidate_mutant.get('id', 'N/A')}' in parent_lookup and no direct target info. Skipping AF3 for this mutant.")

            if not augmented_candidates_for_af3:
                logger.warning(f"No candidates could be augmented with target info. Stopping AL loop.")
                break
            logger.info(f"Successfully augmented {len(augmented_candidates_for_af3)} candidates with target info.")

            # 9. Run TRUE AF3 Evaluation for selected candidates
            newly_evaluated_binders = self._process_binders_batch(augmented_candidates_for_af3, i)
            self.data_manager.update_al_state(newly_evaluated_binders)

            # num_successfully_evaluated = sum(1 for b in newly_evaluated_binders if b.get('af3_status') == 'success')
            # if num_successfully_evaluated == 0 and i < num_al_iterations:
            #     logger.warning(
            #         f"No binders successfully evaluated by AF3 in Iteration {i}. Cannot proceed with further AL.")
            #     break
            final_iteration_num = i
            if i >= 15:
                # final pmpnn redesign
                all_evaluated_binders_df = self.data_manager.get_al_state()
                all_evaluated_binders_data = self.data_manager.load_embeddings_for_binders(
                    all_evaluated_binders_df.to_dict(orient='records')
                )
                candidates_for_final_redesign = [
                    b for b in all_evaluated_binders_data
                    if b.get('af3_status') == 'success' and
                       b.get('is_in_defined_binding_region') and
                       b.get('iptm') >= self.config.filters.complex_filters.min_iptm and
                       b.get('ptm') >= self.config.filters.complex_filters.min_ptm and
                       b.get('complex_plddt') >= self.config.filters.complex_filters.min_complex_plddt and
                       b.get('binder_plddt') >= self.config.filters.complex_filters.min_binder_plddt and
                       b.get('num_interface_resi') >= self.config.filters.complex_filters.num_interface_resi and
                       b.get('pae') <= self.config.filters.complex_filters.min_pae
                    # b.get('shape_complementarity') >= 0.55 and
                    # b.get('interface_delta_unsat_hbonds') <= 3
                ]
                # print(candidates_for_final_redesign)
                if len(candidates_for_final_redesign) >= 250:
                    final_iteration_num = i
                    logger.info(f"Candidates for final pmpnn redesign reached {len(candidates_for_final_redesign)}, stopping AL loop.")
                    break

        logger.info("Full Active Learning loop completed.")

        logger.info("Start final sequence redesign")
        # final pmpnn redesign
        final_iteration_num = final_iteration_num + 1
        all_evaluated_binders_df = self.data_manager.get_al_state()

        top_n = all_evaluated_binders_df.nlargest(300, 'combined_score').to_dict(orient='records')

        candidates_for_final_redesign = candidates_for_final_redesign if candidates_for_final_redesign else top_n

        if candidates_for_final_redesign:
            logger.info(
                f"Selected {len(candidates_for_final_redesign)} top-performing binders (best of each unsolved lineage, above threshold) for ProteinMPNN redesign.")

            redesigned_by_mpnn_raw = self.mpnn_redesigner.redesign_binders_batch(
                binders_to_redesign_data=candidates_for_final_redesign,
                iteration_num=final_iteration_num
            )

            if redesigned_by_mpnn_raw:
                unique_mpnn_sequences_for_af3 = []
                num_mpnn_duplicates_removed = 0
                for mpnn_entry in redesigned_by_mpnn_raw:
                    if mpnn_entry['binder_sequence'] not in self.all_seen_sequences:
                        unique_mpnn_sequences_for_af3.append(mpnn_entry)
                        self.all_seen_sequences.add(mpnn_entry['binder_sequence'])
                    else:
                        num_mpnn_duplicates_removed += 1
                logger.info(
                    f"Removed {num_mpnn_duplicates_removed} duplicate MPNN sequences. Remaining for AF3: {len(unique_mpnn_sequences_for_af3)}")

                if unique_mpnn_sequences_for_af3:
                    logger.info(
                        f"Evaluating {len(unique_mpnn_sequences_for_af3)} ProteinMPNN redesigned binders with AF3.")
                    mpnn_redesigned_and_evaluated = self._process_binders_batch(
                        unique_mpnn_sequences_for_af3, iteration_num=final_iteration_num, keywords='pmpnn'
                    )
                    self._generate_and_store_embeddings(mpnn_redesigned_and_evaluated, final_iteration_num)
                    self.data_manager.update_al_state(mpnn_redesigned_and_evaluated)
                    logger.info(
                        f"Added {len(mpnn_redesigned_and_evaluated)} AF3-evaluated ProteinMPNN binders to AL state.")

        logger.info("Full binder design completed.")

        logger.info("Filtering binders.")
        self.data_manager.filter_binders()
