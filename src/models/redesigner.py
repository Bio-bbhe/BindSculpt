# src/models/redesigner.py
import logging
import os
import time
import subprocess
import json
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from src.utils.structure_utils import find_surface_hydrophobic_residues

# Try to import PyMOL; handle if not available for non-critical parts
try:
    import pymol
    from pymol import cmd as pymol_cmd

    PYMOL_AVAILABLE = True
except ImportError:
    PYMOL_AVAILABLE = False
    pymol_cmd = None  # Placeholder

logger = logging.getLogger(__name__)


class ProteinMPNNRedesigner:
    def __init__(self, config):
        self.config = config.models.proteinmpnn
        self.global_config = config  # Keep a reference to global config for paths

        self.executable_path = self.config.executable_path
        self.helper_script_dir = self.config.helper_script_dir

        self.mpnn_base_temp_dir = os.path.join(config.paths.run_output_dir, "proteinmpnn_runs")
        os.makedirs(self.mpnn_base_temp_dir, exist_ok=True)

        self.enabled = self.config.enabled
        if self.enabled:
            if not PYMOL_AVAILABLE:
                logger.error(
                    "PyMOL is not installed or found. ProteinMPNN redesign (with fixed interface) will not work.")
                self.enabled = False
            elif not os.path.exists(self.executable_path):
                logger.error(f"ProteinMPNN executable not found at: {self.executable_path}. MPNN disabled.")
                self.enabled = False
            else:
                try:
                    # Check for mamba/conda
                    subprocess.run(["mamba", "--version"], check=True, capture_output=True, text=True)
                    logger.info(
                        f"ProteinMPNN Redesigner initialized. Enabled: {self.enabled}.")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.error("Mamba/Conda command not found. ProteinMPNN redesign disabled.")
                    self.enabled = False
        else:
            logger.warning("ProteinMPNN Redesigner is disabled in config.")

    def _count_residues_pymol(self, selection="all"):
        """Helper to count unique residues in a PyMOL selection."""
        if not PYMOL_AVAILABLE: return 0, []
        model = pymol_cmd.get_model(selection)
        residues = set()
        for atom in model.atom:
            residues.add((atom.chain, atom.resi))  # Store as (chain, resi_str)
        # Convert resi to int for sorting, handle potential non-numeric resi if any
        contact_resn_int = sorted([int(r[1]) for r in residues if r[1].isdigit()])
        pymol_cmd.delete("all")
        return len(residues), contact_resn_int

    def _get_interface_fixed_positions_pymol(self, pdb_file_path: str, chain_to_design: str = "B",
                                             interacting_chain: str = "A"):
        """
        Uses PyMOL to identify interface residues on the chain_to_design
        that are interacting with interacting_chain.
        Returns a list of residue numbers (1-indexed) to be fixed.
        """
        print('running _get_interface_fixed_positions_pymol')
        if not PYMOL_AVAILABLE:
            logger.warning("PyMOL not available. Cannot determine fixed interface positions. Designing full chain.")
            return []

        try:
            # pymol.finish_launching(['pymol', '-cq'])  # Quiet, command-line mode
            protein_acc = Path(pdb_file_path).stem
            pymol_cmd.load(pdb_file_path, protein_acc)

            selection_design = f"chain {chain_to_design}"
            selection_target = f"chain {interacting_chain}"

            pymol_cmd.select(selection_design, f"chain {chain_to_design}")
            pymol_cmd.select(selection_target, f"chain {interacting_chain}")

            # Select residues in chain_to_design that are close to interacting_chain
            interaction_selection_name = f"interface_residues_on_{chain_to_design}"
            distance_threshold = self.config.fixed_pos_distance_threshold

            pymol_cmd.select(
                interaction_selection_name,
                f"byres ({selection_design} within {distance_threshold} of {selection_target})"
            )

            _, contact_resn_design_chain = self._count_residues_pymol(interaction_selection_name)

            pymol_cmd.delete("all")  # Clean up

            return contact_resn_design_chain
        except Exception as e:
            logger.error(f"Error during PyMOL interface calculation for {pdb_file_path}: {e}")
            return []  # Return empty list on error, so MPNN designs whole chain

    def _run_mpnn_subprocess(self, cmd_list: list, run_identifier: str):
        """Helper to run a subprocess command for MPNN."""
        try:
            logger.info(f"Executing MPNN step for {run_identifier}. Command: {' '.join(cmd_list)}")
            process = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
            logger.debug(f"MPNN step stdout for {run_identifier}:\n{process.stdout}")
            logger.debug(f"MPNN step stderr for {run_identifier}:\n{process.stderr}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(
                f"MPNN step failed for {run_identifier} (Exit Code: {e.returncode}). Stdout: {e.stdout}. Stderr: {e.stderr}")
        except FileNotFoundError:
            logger.error(
                f"'{cmd_list[0]}' command not found for MPNN step '{run_identifier}'. Check PATH or script paths.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during MPNN step for {run_identifier}: {e}")
        return False

    def _select_top_n_mpnn_seqs(self, mpnn_fasta_output_path: str, num_seqs: int):
        """Selects top N sequences from a ProteinMPNN output FASTA based on score."""
        if not os.path.exists(mpnn_fasta_output_path):
            logger.warning(f"MPNN FASTA output not found: {mpnn_fasta_output_path}")
            return []

        redesigned_sequences = []
        try:
            records_dict = {}
            score2key = {}

            # ProteinMPNN output FASTA has scores in description line
            # >name, score=X, global_score=Y, ...
            # >T=0.1, sample=1, score=Z, ...
            for record in SeqIO.parse(mpnn_fasta_output_path, "fasta"):
                records_dict[record.description] = str(record.seq)
                # Parse score (usually the first score in the description is relevant, or 'score=')
                # This might need refinement based on exact ProteinMPNN output format
                try:
                    # Attempt to extract score field
                    score_str = record.description.split("score=")[1].split(",")[0]
                    score = float(score_str)
                    score2key[record.description] = score
                except (IndexError, ValueError):
                    logger.warning(
                        f"Could not parse score from description: {record.description} in {mpnn_fasta_output_path}")

            # Sort by score (ProteinMPNN scores are often log-probabilities, lower is better, but check your MPNN version)
            # Your example sorts ascending: sorted(score2key.items(), key=lambda item: item[1]) -> lower is better
            sorted_items = sorted(score2key.items(), key=lambda item: item[1])

            selected_count = 0
            for desc_key, score_val in sorted_items:
                if selected_count < num_seqs:
                    redesigned_sequences.append({
                        'sequence': records_dict[desc_key],
                        'mpnn_score': score_val,
                        'mpnn_description': desc_key
                    })
                    selected_count += 1
                else:
                    break
            logger.info(f"Selected {len(redesigned_sequences)} top sequences from {mpnn_fasta_output_path}")

        except Exception as e:
            logger.error(f"Error parsing MPNN FASTA {mpnn_fasta_output_path}: {e}")
        return redesigned_sequences

    def redesign_binders_batch(self, binders_to_redesign_data: list, iteration_num: int):
        """
        Redesigns a batch of binder sequences using ProteinMPNN.

        Args:
            binders_to_redesign_data (list): List of binder dictionaries. Each must have
                                             'id', 'af3_pdb_path', 'target_id', 'target_sequence'.
            iteration_num (int): Current AL iteration number for organizing outputs.

        Returns:
            list: A list of new binder dictionaries for the redesigned sequences,
                  ready for AF3 evaluation.
        """
        if not self.enabled:
            logger.debug("ProteinMPNN is disabled. Skipping redesign batch.")
            return []
        if not binders_to_redesign_data:
            logger.warning("No binders provided for ProteinMPNN redesign.")
            return []

        batch_run_identifier = f"mpnn_iter_{iteration_num}_{int(time.time())}"
        current_batch_temp_dir = os.path.join(self.mpnn_base_temp_dir, batch_run_identifier)
        os.makedirs(current_batch_temp_dir, exist_ok=True)

        # Directory to store input PDBs for this batch
        batch_input_pdb_dir = os.path.join(current_batch_temp_dir, "input_pdbs")
        os.makedirs(batch_input_pdb_dir, exist_ok=True)

        # Directory where ProteinMPNN will write its sequence outputs
        batch_mpnn_design_out_dir = os.path.join(current_batch_temp_dir, "designed_seqs")
        os.makedirs(batch_mpnn_design_out_dir, exist_ok=True)

        # 1. Prepare input PDBs by copying them to a central location for this batch
        #    and creating the parsed_pdbs.jsonl
        parsed_pdbs_jsonl_path = os.path.join(current_batch_temp_dir, "parsed_pdbs.jsonl")

        # Symlink or copy PDBs to batch_input_pdb_dir
        # pymol.finish_launching(['pymol', '-cq'])
        for binder_entry in binders_to_redesign_data:
            pdb_path_original = binder_entry.get('af3_pdb_path')
            if pdb_path_original and os.path.exists(pdb_path_original):
                # Use a consistent naming scheme for MPNN processing based on binder_entry['id']
                pdb_file_path = f"{binder_entry['id']}.pdb"
                pdb_path_for_pmpnn = os.path.join(batch_input_pdb_dir, pdb_file_path)
                try:
                    pymol_cmd.load(pdb_path_original)
                    object_name = pymol_cmd.get_object_list('(all)')[0]
                    pymol_cmd.extract("cleaned_object", object_name)
                    pymol_cmd.save(pdb_path_for_pmpnn,"cleaned_object")
                    binder_entry['mpnn_input_pdb'] = pdb_path_for_pmpnn  # Store path for later
                    pymol_cmd.delete("all")  # Clean up
                except Exception as e:
                    logger.error(f"Failed to copy PDB {pdb_path_original} for MPNN: {e}")
                    binder_entry['mpnn_status'] = "failed_pdb_copy"
            else:
                logger.warning(f"PDB path missing or invalid for binder {binder_entry['id']}. Skipping MPNN.")
                binder_entry['mpnn_status'] = "missing_pdb"

        # Step 1: Parse multiple chains (run on the directory of copied PDBs)
        cmd1 = [
            "python",
            os.path.join(self.helper_script_dir, "parse_multiple_chains.py"),
            f"--input_path={batch_input_pdb_dir}",  # Directory with PDBs
            f"--output_path={parsed_pdbs_jsonl_path}"
        ]
        if not self._run_mpnn_subprocess(cmd1, f"{batch_run_identifier}_step1_parse_chains"):
            return []  # Critical failure

        # Step 2: Assign fixed chains (target chain 'A')
        assigned_pdbs_jsonl_path = os.path.join(current_batch_temp_dir, "assigned_pdbs.jsonl")
        # Convert list of chains to string for command line
        cmd2 = [
            "python",
            os.path.join(self.helper_script_dir, "assign_fixed_chains.py"),
            "--input_path", parsed_pdbs_jsonl_path,
            "--output_path", assigned_pdbs_jsonl_path,
            "--chain_list", self.config.chains_to_design[0]  # e.g., "A"
        ]
        if not self._run_mpnn_subprocess(cmd2, f"{batch_run_identifier}_step2_assign_fixed_chains"):
            return []

        # Step 3: Assign fixed positions (interface residues on binder chain 'B')
        # We need to generate the fixed_positions.jsonl file
        fixed_positions_jsonl_path = os.path.join(current_batch_temp_dir, "fixed_positions.jsonl")
        fixed_positions_dict_for_jsonl = {}
        # is in AL loop or the final redesign
        al_design = (iteration_num < self.global_config.active_learning.num_iterations + 1)
        iteration_num = iteration_num if al_design else iteration_num + 1
        for binder_entry in binders_to_redesign_data:
            if binder_entry.get('mpnn_status', '').startswith('failed') or 'mpnn_input_pdb' not in binder_entry:
                continue  # Skip if PDB was missing or copy failed

            pdb_name_key = Path(binder_entry['mpnn_input_pdb']).stem  # e.g., binder_entry['id']

            if al_design:  # for AL loop redesign
                fixed_residues_chain_b = self._get_interface_fixed_positions_pymol(
                    binder_entry['mpnn_input_pdb'],
                    chain_to_design=self.config.chains_to_design[0]  # Assuming single chain like "B"
                )
            else:  # for final design
                fixed_residues_chain_b = find_surface_hydrophobic_residues(
                    pdb_file=binder_entry['mpnn_input_pdb'],
                    chain_id=self.config.chains_to_design[0],  # Assuming single chain like "B"
                    threshold=self.config.relative_solvent_accessibility
                )
                logger.info(f"Final pmpnn redesign, fixed positions JSONL created at: {fixed_positions_jsonl_path}")

            # ProteinMPNN fixed_positions_jsonl expects:
            # {"pdb_name": {"A": [1,2,3], "B": [10,11,12]}, ...}
            # We only fix positions on the chain being designed (e.g., "B")
            fixed_pos_entry = {chain_id: [] for chain_id in
                               self.config.chains_to_keep_fixed_for_mpnn + self.config.chains_to_design}
            fixed_pos_entry[self.config.chains_to_design[0]] = fixed_residues_chain_b
            fixed_positions_dict_for_jsonl[pdb_name_key] = fixed_pos_entry

        try:
            with open(fixed_positions_jsonl_path, 'w') as f:
                json.dump(fixed_positions_dict_for_jsonl, f)
            logger.info(f"Fixed positions JSONL created at: {fixed_positions_jsonl_path}")
        except Exception as e:
            logger.error(f"Failed to write fixed_positions.jsonl: {e}")
            return []
        # Step 4: Run ProteinMPNN
        cmd4 = [
            "python",
            self.executable_path,
            "--jsonl_path", parsed_pdbs_jsonl_path,
            "--chain_id_jsonl", assigned_pdbs_jsonl_path,
            "--fixed_positions_jsonl", fixed_positions_jsonl_path,
            "--out_folder", batch_mpnn_design_out_dir,
            "--num_seq_per_target", str(self.config.num_seq_per_target_pdb),
            "--seed", "42",  # Or make configurable
            "--batch_size", str(self.config.num_seq_per_target_pdb),  # Batch per PDB
            "--use_soluble_model",  # Assuming this is desired
            "--model_name", "v_48_020",  # Or make configurable
            "--sampling_temp", str(self.config.sampling_temp) if al_design else '0.5',  # final redesign using 0.5
        ]
        if not self._run_mpnn_subprocess(cmd4, f"{batch_run_identifier}_step4_run_mpnn"):
            return []  # Critical failure

        # 5. Collect and process results
        all_redesigned_binders_for_af3 = []
        mpnn_output_seqs_dir = os.path.join(batch_mpnn_design_out_dir, "seqs")

        for original_binder_entry in binders_to_redesign_data:
            if original_binder_entry.get('mpnn_status', '').startswith(
                    'failed') or 'mpnn_input_pdb' not in original_binder_entry:
                continue

            # The FASTA file is named after the PDB stem used in MPNN
            mpnn_fasta_filename = f"{Path(original_binder_entry['mpnn_input_pdb']).stem}.fa"
            mpnn_fasta_path = os.path.join(mpnn_output_seqs_dir, mpnn_fasta_filename)

            top_redesigned_seq_data = self._select_top_n_mpnn_seqs(mpnn_fasta_path,
                                                                   self.config.num_top_mpnn_seqs_to_evaluate)

            for i, redesigned_data in enumerate(top_redesigned_seq_data):
                new_binder_entry = {
                    'id': f"{original_binder_entry['id']}N{i + 1}",
                    'binder_sequence': redesigned_data['sequence'],
                    'parent_id': original_binder_entry['id'],
                    'original_parent_id': f"{original_binder_entry['id']}N{iteration_num}s{i + 1}",
                    'target_id': original_binder_entry['target_id'],
                    'target_sequence': original_binder_entry['target_sequence'],
                    'mutation_type': 'proteinmpnn_redesign',
                    'mpnn_score': redesigned_data['mpnn_score'],
                    'mpnn_description': redesigned_data['mpnn_description'],
                    'iteration': iteration_num,  # Mark as current iteration
                    # Other fields like af3_status, iptm, embedding_path will be populated after AF3
                }
                all_redesigned_binders_for_af3.append(new_binder_entry)

        logger.info(
            f"ProteinMPNN redesign generated {len(all_redesigned_binders_for_af3)} new candidates for AF3 evaluation.")
        # pymol_cmd.quit()
        return all_redesigned_binders_for_af3