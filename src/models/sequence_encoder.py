# src/models/sequence_encoder.py
import logging
import torch
import numpy as np
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm
import os  # Added for file operations

logger = logging.getLogger(__name__)


class SequenceEncoder:
    def __init__(self, config):
        self.config = config.models.esm_encoder
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings_output_dir = config.paths.embeddings_dir  # Get embeddings output directory
        os.makedirs(self.embeddings_output_dir, exist_ok=True)  # Ensure it exists
        logger.info(
            f"SequenceEncoder will use device: {self.device}. Embeddings will be saved to: {self.embeddings_output_dir}")

    def load_model(self):
        """Loads the ESM tokenizer and model."""
        model_location = self.config.model_location
        logger.info(f"Loading ESM tokenizer and model from: {model_location}")
        try:
            self.tokenizer = EsmTokenizer.from_pretrained(model_location)
            self.model = EsmModel.from_pretrained(model_location).to(self.device)
            self.model.eval()
            logger.info("ESM model loaded successfully.")
            # --- NEW DEBUG ---
            logger.debug(f"DEBUG: self.model is set: {self.model is not None}")
            logger.debug(f"DEBUG: self.tokenizer is set: {self.tokenizer is not None}")
            # --- END DEBUG ---
        except Exception as e:
            logger.error(f"Failed to load ESM model or tokenizer from {model_location}: {e}")
            # --- CRITICAL: Ensure model and tokenizer are explicitly set to None on failure ---
            self.model = None
            self.tokenizer = None
            # --- END CRITICAL ---
            raise # Re-raise the exception to be caught by the orchestrator's init

    def get_embeddings(self, sequences_data: list, iteration_num: int):  # Added iteration_num
        """
        Computes ESM embeddings for a list of sequences and saves them as .npy files.

        Args:
            sequences_data (list): List of dictionaries, e.g.,
                                   [{'id': 'seq_A', 'sequence': 'ACDEF'}, ...]
            iteration_num (int): Current active learning iteration number, for organizing files.

        Returns:
            list: List of dictionaries, with original data + 'embedding_path' (path to .npy file).
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("ESM model not loaded. Call load_model() first.")
        if not sequences_data:
            logger.warning("No sequences provided for embedding generation. Returning empty list.")
            return []

        batch_size = self.config.batch_size
        truncation_seq_length = self.config.truncation_seq_length

        embeddings_results = []

        # Create iteration-specific subdirectory for embeddings
        iter_embeddings_dir = os.path.join(self.embeddings_output_dir, f"iteration_{iteration_num}")
        os.makedirs(iter_embeddings_dir, exist_ok=True)

        seq_ids = [s['id'] for s in sequences_data]
        seq_strs = [s['sequence'] for s in sequences_data]

        logger.info(f"Generating embeddings for {len(sequences_data)} sequences.")

        for i in tqdm(range(0, len(seq_strs), batch_size), desc="ESM Embeddings", unit="batch"):
            batch_ids = seq_ids[i:i + batch_size]
            batch_seqs = seq_strs[i:i + batch_size]

            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=truncation_seq_length + 2,
                add_special_tokens=True,
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.last_hidden_state

            for j in range(hidden_states.size(0)):
                seq_len = torch.sum(attention_mask[j]) - 2
                seq_embeddings = hidden_states[j, 1:1 + seq_len, :]

                mean_embedding = seq_embeddings.mean(dim=0).cpu().numpy()

                # Save embedding to .npy file
                embedding_filename = f"{batch_ids[j]}.npy"
                embedding_filepath = os.path.join(iter_embeddings_dir, embedding_filename)
                try:
                    np.save(embedding_filepath, mean_embedding)
                    embeddings_results.append({
                        'id': batch_ids[j],
                        'sequence': batch_seqs[j],
                        'embedding_path': embedding_filepath  # Store path instead of array
                    })
                except Exception as e:
                    logger.error(f"Failed to save embedding for {batch_ids[j]} to {embedding_filepath}: {e}")
                    # If saving fails, don't add path, mark as None
                    embeddings_results.append({
                        'id': batch_ids[j],
                        'sequence': batch_seqs[j],
                        'embedding_path': None
                    })
        logger.info(f"Completed embedding generation for {len(embeddings_results)} sequences.")
        return embeddings_results