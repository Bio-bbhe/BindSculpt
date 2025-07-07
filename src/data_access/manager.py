# src/data_access/manager.py
import pandas as pd
import os
import logging
from Bio import SeqIO
import numpy as np # Added for loading .npy files

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self, config):
        self.config = config
        self.inputs_dir = config.paths.data_inputs
        self.target_fasta = config.paths.target_fasta
        self.outputs_dir = config.paths.run_output_dir
        self.al_state_path = os.path.join(self.outputs_dir, "al_state.csv") # Or .sqlite
        self.al_iteration_state_path = os.path.join(self.outputs_dir, "al_iteration_state.csv")
        self.filtered_state_path = os.path.join(self.outputs_dir, "al_filtered_state.csv")
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(os.path.join(self.outputs_dir, config.models.alphafold3.output_subdir), exist_ok=True) # Ensure AF3 output dir exists
        os.makedirs(config.paths.embeddings_dir, exist_ok=True) # Ensure embeddings dir exists
        os.makedirs(config.paths.surrogate_models_dir, exist_ok=True) # Ensure surrogate models dir exists

    def load_initial_targets(self):
        # ... (no change to this method) ...
        targets_list = []

        if not os.path.exists(self.target_fasta):
            logger.warning(f"Target file not found at {self.target_fasta}. Returning dummy target.")
            return [{"id": "target_01", "sequence": "MGDSSKSDGDYLDAGKKSLDGSDGDYLDAGKKSLDGSDGDYLDAGKKSLDGSDGDYLDAGKKSLDG"}]

        try:
            for record in SeqIO.parse(self.target_fasta, "fasta"):
                targets_list.append({'id': record.id, 'sequence': str(record.seq)})
            logger.info(f"Loaded {len(targets_list)} initial targets from {self.target_fasta} using Biopython.")
        except Exception as e:
            logger.error(f"Error parsing FASTA file {self.target_fasta} with Biopython: {e}")
            raise

        return targets_list

    def save_iteration_results(self, iteration_num, binders_data,keywords='no_pmpnn'):
        # ... (no change to this method, it saves whatever is in binders_data list) ...
        # Ensure embeddings (if present) are stored as paths, not raw arrays in the dicts here
        # This is handled by _generate_and_store_embeddings in loop.py now
        df = pd.DataFrame(binders_data)
        filepath = os.path.join(self.outputs_dir, f"iteration_{iteration_num}_{keywords}_results.csv")
        df.to_csv(filepath, index=False)
        logger.info(f"Saved iteration {iteration_num} results to {filepath}")

    def get_al_state(self):
        if not os.path.exists(self.al_state_path) or os.path.getsize(self.al_state_path) == 0:
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.al_state_path)
            if not df.empty and 'embedding_path' in df.columns:
                df['embedding_path'] = df['embedding_path'].fillna('').astype(str)
                return df
        except pd.errors.EmptyDataError:
            logger.error(f"al_state.csv is empty: {e}")

        return pd.DataFrame()

    def update_al_state(self, new_data):
        # ... (no change, updates CSV with paths) ...
        current_state = self.get_al_state()
        new_df = pd.DataFrame(new_data)
        updated_state = pd.concat([current_state, new_df], ignore_index=True)
        updated_state.to_csv(self.al_state_path, index=False)
        logger.info(f"Updated overall AL state with {len(new_df)} new entries.")

    def load_embedding_from_path(self, embedding_path):
        """Helper to load a single embedding from an .npy file."""
        if embedding_path and os.path.exists(embedding_path):
            try:
                return np.load(embedding_path)
            except Exception as e:
                logger.error(f"Failed to load embedding from {embedding_path}: {e}")
        return None

    def load_embeddings_for_binders(self, binders_data_list):
        """
        Loads embeddings for a list of binders that have 'embedding_path' keys.
        Updates the 'embedding' key with the loaded NumPy array.
        """
        for binder_entry in binders_data_list:
            if 'embedding_path' in binder_entry and binder_entry['embedding_path']:
                binder_entry['embedding'] = self.load_embedding_from_path(binder_entry['embedding_path'])
            else:
                binder_entry['embedding'] = None # Ensure embedding is None if path is missing/invalid
        return binders_data_list

    def filter_binders(self):
        al_df = self.get_al_state()
        df = al_df[
            (al_df['af3_status'] == 'success') &
            (al_df['iptm'] >= self.config.filters.complex_filters.min_iptm) &
            (al_df['ptm'] >= self.config.filters.complex_filters.min_ptm) &
            (al_df['complex_plddt'] >= self.config.filters.complex_filters.min_complex_plddt) &
            (al_df['binder_plddt'] >= self.config.filters.complex_filters.min_binder_plddt) &
            (al_df['pae'] <= self.config.filters.complex_filters.min_pae)
            ]

        df.to_csv(self.al_iteration_state_path , index=False)

        df = df[
            (df['shape_complementarity'] >= self.config.filters.complex_filters.min_shape_complementarity) &
            (df['surface_hydrophobicity'] <= self.config.filters.complex_filters.max_surface_hydrophobicity) &
            (df['interface_delta_unsat_hbonds'] <= self.config.filters.complex_filters.max_interface_delta_unsat_hbonds)
            ]

        df.to_csv(self.filtered_state_path, index=False)
        logger.info(f"Saved results to {self.filtered_state_path}")
