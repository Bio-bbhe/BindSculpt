# src/models/sequence_encoder.py
import logging
import torch
import numpy as np
import pandas as pd
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

    def _save_with_verification(self, embedding: np.ndarray, filepath: str, retries=2) -> bool:
        """Saves embedding with verification and retry logic"""
        for attempt in range(retries + 1):
            try:
                # Save embedding
                np.save(filepath, embedding)

                # Verify file existence and contents
                if self._verify_embedding_file(filepath, embedding):
                    return True

                logger.warning(f"Verification failed on attempt {attempt + 1} for {filepath}")

            except Exception as e:
                logger.warning(f"Save exception on attempt {attempt + 1} for {filepath}: {e}")

            # Wait before retry
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))

        return False

    def _verify_embedding_file(self, filepath: str, original_embedding: np.ndarray) -> bool:
        """Verifies the saved embedding file matches the original"""
        try:
            # Check file exists and has content
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return False

            if os.path.getsize(filepath) == 0:
                logger.error(f"Empty file: {filepath}")
                return False

            # Load and verify contents
            loaded = np.load(filepath)

            # Shape verification
            if loaded.shape != original_embedding.shape:
                logger.error(
                    f"Shape mismatch: Expected {original_embedding.shape}, "
                    f"got {loaded.shape} in {filepath}"
                )
                return False

            # Value verification (tolerance for float precision)
            if not np.allclose(loaded, original_embedding, atol=1e-6):
                logger.error(
                    f"Value mismatch in {filepath}. "
                    f"Max diff: {np.max(np.abs(loaded - original_embedding))}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Verification failed for {filepath}: {e}")
            return False

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

        # --- KEY CODE TO SAVE DATA USING DF ---
        debug_csv_path = os.path.join(self.embeddings_output_dir, f"debug_input_sequences_iter_{iteration_num}.csv")
        logger.info(f"Saving incoming sequence data for inspection to: {debug_csv_path}")

        # Convert list of dicts to DataFrame and save to CSV
        pd.DataFrame(sequences_data).to_csv(debug_csv_path, index=False)
        # --- END OF KEY CODE ---

        seq_ids = [s['id'] for s in sequences_data]
        seq_strs = [s['sequence'] for s in sequences_data]
        # print(f'sequence data is {sequences_data}')
        logger.info(f"Generating embeddings for {len(sequences_data)} sequences.")

        for i in tqdm(range(0, len(seq_strs), batch_size), desc="ESM Embeddings", unit="batch"):
            batch_ids = seq_ids[i:i + batch_size]
            batch_seqs = seq_strs[i:i + batch_size]

            try:
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

                # Check if model output size matches batch input size
                if hidden_states.size(0) != len(batch_ids):
                    logger.error(f"CRITICAL BATCH MISMATCH: Batch input size was {len(batch_ids)}, "
                                 f"but ESM model output size was {hidden_states.size(0)}. Skipping this batch.")
                    # Mark all in this batch as failed
                    for k in range(len(batch_ids)):
                        embeddings_results.append(
                            {'id': batch_ids[k], 'sequence': batch_seqs[k], 'embedding_path': None})
                    continue  # Move to next batch

            except Exception as e:
                logger.error(
                    f"CRITICAL ERROR during tokenization or ESM model forward pass for batch starting with ID {batch_ids[0]}: {e}")
                # Mark all sequences in this batch as failed
                for k in range(len(batch_ids)):
                    embeddings_results.append({'id': batch_ids[k], 'sequence': batch_seqs[k], 'embedding_path': None})
                continue  # Skip to next batch on error
            # inputs = self.tokenizer(
            #     batch_seqs,
            #     return_tensors="pt",
            #     padding=True,
            #     truncation=True,
            #     max_length=truncation_seq_length + 2,
            #     add_special_tokens=True,
            # )
            # input_ids = inputs["input_ids"].to(self.device)
            # attention_mask = inputs["attention_mask"].to(self.device)
            #
            # with torch.no_grad():
            #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            #     hidden_states = outputs.last_hidden_state

            for j in range(hidden_states.size(0)):
                seq_len = torch.sum(attention_mask[j]) - 2
                seq_embeddings = hidden_states[j, 1:1 + seq_len, :]

                mean_embedding = seq_embeddings.mean(dim=0).cpu().numpy()

                # Save embedding to .npy file
                embedding_filename = f"{batch_ids[j]}.npy"
                embedding_filepath = os.path.join(iter_embeddings_dir, embedding_filename)
                try:
                    # === NEW: VERIFICATION AFTER SAVING ===
                    success = self._save_with_verification(mean_embedding, embedding_filepath)

                    if success:
                        embeddings_results.append({
                            'id': batch_ids[j],
                            'sequence': batch_seqs[j],
                            'embedding_path': embedding_filepath
                        })
                    else:
                        raise RuntimeError("Embedding file verification failed")

                except Exception as e:
                    logger.error(
                        f"Failed to save/verify embedding for {batch_ids[j]}: {e}\n"
                        f"Path: {embedding_filepath}\n"
                        f"Embedding shape: {mean_embedding.shape} | dtype: {mean_embedding.dtype}"
                    )
                    # If saving fails, don't add path, mark as None
                    embeddings_results.append({
                        'id': batch_ids[j],
                        'sequence': batch_seqs[j],
                        'embedding_path': None
                    })
        # === NEW: FINAL FILE COUNT VERIFICATION ===
        saved_files = os.listdir(iter_embeddings_dir)
        saved_count = len(saved_files)
        if saved_count < len(sequences_data):
            missing = len(sequences_data) - saved_count
            logger.critical(
                f"EMBEDDING FILE COUNT MISMATCH: Expected {len(sequences_data)} files, "
                f"saved {saved_count}. Missing {missing} files!"
            )
            # Log which specific sequences are missing
            saved_ids = {os.path.splitext(f)[0] for f in saved_files}
            all_ids = {s['id'] for s in sequences_data}
            missing_ids = all_ids - saved_ids
            logger.critical(f"Missing embeddings for IDs: {sorted(missing_ids)}")
            raise RuntimeError("Embedding file verification failed")
        else:
            logger.info(f"Successfully saved all {saved_count} embedding files")
        logger.info(f"Completed embedding generation for {len(embeddings_results)} sequences.")
        return embeddings_results