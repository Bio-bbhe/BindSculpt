import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EncoderDecoderModel
import time
from tqdm import tqdm
import pytorch_lightning as pl  # Keep for seed_everything, but don't need the Trainer

# Import the helper functions
from src.utils.helpers import compute_identity, get_bad_word_ids_for_tokenizer

logger = logging.getLogger(__name__)


class ProteinSeqDataset(Dataset):
    """
    Dataset for loading target sequences for binder generation.
    Adapted to take a list of (target_id, target_sequence) tuples directly,
    instead of parsing from a file path.
    """

    def __init__(self, target_data, tokenizer, encoder_max_length=1000):
        """
        Args:
            target_data (list): A list of dictionaries, each with 'id' and 'sequence'.
            tokenizer: The pre-trained tokenizer.
            encoder_max_length (int): Max length for encoder input.
        """
        self.tokenizer = tokenizer
        self.encoder_max_length = encoder_max_length
        self.target_data = target_data  # [{ 'id': 'target_A', 'sequence': 'MGD...' }, ...]

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, idx):
        target_entry = self.target_data[idx]
        target_id = target_entry['id']
        original_target_seq = target_entry['sequence']  # Renamed for clarity

        # Use the pre-processed input sequence, which might be a sub-region
        input_for_tokenizer = target_entry['input_target_region_sequence']

        # Tokenize the target sequence
        inputs = self.tokenizer(input_for_tokenizer, return_tensors='pt',  # Use the correct sequence variable
                                max_length=self.encoder_max_length,
                                truncation=True,
                                padding='max_length')
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'target_seq': original_target_seq,  # Pass original full sequence for logging/results
            'input_target_region_sequence': input_for_tokenizer,  # Pass the actual input sequence
            'target_id': target_id
        }


class BinderGenerator:
    def __init__(self, config):
        self.global_config = config
        self.config = config.models.seq2seq
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Set PyTorch Lightning seed globally, once
        pl.seed_everything(seed=self.config.generation_params.seed)

    def load_model(self):
        """Loads the pre-trained tokenizer and EncoderDecoder model."""
        tokenizer_path = self.config.tokenizer_path
        model_path = self.config.model_path

        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        logger.info(f"Loading model from: {model_path}")
        self.model = EncoderDecoderModel.from_pretrained(model_path).to(self.device)
        self.model.eval()  # Set to evaluation mode

        self.bad_words_ids = get_bad_word_ids_for_tokenizer(self.tokenizer)
        logger.info(f"Loaded Seq2Seq model and tokenizer. Bad words IDs: {self.bad_words_ids[:5]}...")

    # Helper function to parse binding region string
    def _parse_binding_region(self, region_str):
        """
        Parses a string like "1,5,10-15" into a sorted list of 0-based indices.
        Returns a sorted list of unique indices.
        """
        indices = set()
        parts = region_str.split(',')
        for part in parts:
            part = part.strip()
            if not part: continue
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    # Add all indices in the range (inclusive), converting from 1-based to 0-based
                    for i in range(start, end + 1):
                        indices.add(i - 1)
                except ValueError:
                    logger.warning(f"Could not parse range '{part}' in binding_region. Skipping.")
            else:
                try:
                    # Add single index, converting from 1-based to 0-based
                    indices.add(int(part) - 1)
                except ValueError:
                    logger.warning(f"Could not parse index '{part}' in binding_region. Skipping.")

        return sorted(list(indices))

    def generate_binders(self, targets_to_generate, num_return_sequences=None):
        """
        Generates binder sequences for a list of target proteins.

        Args:
            targets_to_generate (list): A list of dicts, e.g., [{'id': 'target_A', 'sequence': 'MGD...'}]
            num_return_sequences (int, optional): Number of binders to generate per target.
                                                  Defaults to config value.

        Returns:
            list: A list of dictionaries, each representing a generated binder
                  with its metadata and generation parameters.
        """
        if not self.model or not self.tokenizer:
            logger.error("Seq2Seq model or tokenizer not loaded. Call load_model() first.")
            raise RuntimeError("Model or tokenizer not loaded.")

        if not targets_to_generate:
            logger.warning("No targets provided for generation. Returning empty list.")
            return []

        # Override num_return_sequences if specified, otherwise use config
        num_return_sequences = num_return_sequences if num_return_sequences is not None else self.config.generation_params.num_return_sequences

        # Process targets to handle binding regions ---
        processed_targets_to_generate = []
        # Check for binding region definition in config
        use_binding_region = self.config.define_binding.get('enabled', False)
        binding_region_str = self.config.define_binding.get('binding_region')

        indices_to_extract = None
        if use_binding_region and binding_region_str:
            logger.info(f"Activating binding region targeting with region: {binding_region_str}")
            indices_to_extract = self._parse_binding_region(binding_region_str)

        for target in targets_to_generate:
            processed_target = target.copy()  # Work on a copy
            original_seq = target['sequence']

            if indices_to_extract:
                # Build the new sequence from specified indices
                region_seq_list = [original_seq[i] for i in indices_to_extract if i < len(original_seq)]
                input_seq = "".join(region_seq_list)
                processed_target['input_target_region_sequence'] = input_seq
                logger.debug(f"For target {target['id']}, using region sequence of length {len(input_seq)}")
            else:
                # If no region is defined, the input sequence is the original sequence
                processed_target['input_target_region_sequence'] = original_seq

            processed_targets_to_generate.append(processed_target)

        # Create dataset and data loader for the given targets
        dataset = ProteinSeqDataset(
            target_data=processed_targets_to_generate,  # Use the processed list
            tokenizer=self.tokenizer,
            encoder_max_length=self.config.generation_params.encoder_max_length
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.generation_params.batch_size,
            shuffle=False,  # No need to shuffle for inference
            num_workers=self.config.generation_params.num_workers
        )

        results = []
        overall_start_time = time.time()

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Generating Binders", unit="batch"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_seqs_batch = batch['target_seq']
                input_region_seqs_batch = batch[
                    'input_target_region_sequence']  # Get the region sequence from the batch
                target_ids_batch = batch['target_id']

                # Iterate through different sampling strategies defined in config
                sampling_strategys = self.config.sampling_strategies
                if isinstance(sampling_strategys, str):  # Handle comma-separated string if config is not list
                    sampling_strategys = [s.strip() for s in sampling_strategys.split(',')]
                elif not isinstance(sampling_strategys, list):
                    logger.warning(
                        f"Invalid sampling_strategies in config: {sampling_strategys}. Defaulting to 'top_k'.")
                    sampling_strategys = ["top_k"]

                for sampling_strategy in sampling_strategys:
                    method_start_time = time.time()
                    gen_params = self.config.generation_params

                    common_generate_kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "max_length": gen_params.max_length,
                        "min_length": gen_params.min_length,
                        "bos_token_id": self.tokenizer.cls_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "bad_words_ids": self.bad_words_ids,
                        "num_return_sequences": num_return_sequences,
                        "repetition_penalty": gen_params.repetition_penalty,
                        "no_repeat_ngram_size": gen_params.no_repeat_ngram_size
                    }

                    if sampling_strategy == "beam":
                        generated_outputs = self.model.generate(
                            **common_generate_kwargs,
                            num_beams=gen_params.num_beams,
                            num_beam_groups=gen_params.num_beam_groups,
                            early_stopping=True,
                            diversity_penalty=gen_params.diversity_penalty
                        )
                        params_str = (
                            f"beam_numBeams_{gen_params.num_beams}_numBeamGroups_{gen_params.num_beam_groups}_"
                            f"diversityPenalty_{gen_params.diversity_penalty}_maxlength_{gen_params.max_length}_"
                            f"minlength_{gen_params.min_length}")
                    elif sampling_strategy == "top_k":
                        generated_outputs = self.model.generate(
                            **common_generate_kwargs,
                            do_sample=True,
                            top_k=gen_params.top_k,
                            num_beams=1,  # For sampling, num_beams should be 1
                            early_stopping=False,
                            temperature=gen_params.temperature
                        )
                        params_str = (f"topK_topK_{gen_params.top_k}_temperature_{gen_params.temperature}_"
                                      f"maxlength_{gen_params.max_length}_minlength_{gen_params.min_length}")
                    elif sampling_strategy == "top_p":
                        generated_outputs = self.model.generate(
                            **common_generate_kwargs,
                            do_sample=True,
                            top_p=gen_params.top_p,
                            num_beams=1,  # For sampling, num_beams should be 1
                            early_stopping=False,
                            temperature=gen_params.temperature
                        )
                        params_str = (f"topP_topP_{gen_params.top_p}_temperature_{gen_params.temperature}_"
                                      f"maxlength_{gen_params.max_length}_minlength_{gen_params.min_length}")
                    else:
                        logger.warning(f"Unknown sampling strategy: {sampling_strategy}. Skipping.")
                        continue  # Skip to next strategy

                    method_elapsed_time = time.time() - method_start_time

                    decoded_sequences = [seq.replace(" ", "") for seq in
                                         self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)]

                    # Map generated sequences back to their original target inputs
                    # Each input in the batch produces num_return_sequences
                    for i in range(len(target_ids_batch)):
                        target_id = target_ids_batch[i]
                        target_seq = target_seqs_batch[i]
                        input_region_seq = input_region_seqs_batch[i]  # Get corresponding region sequence

                        start_idx = i * num_return_sequences
                        end_idx = (i + 1) * num_return_sequences
                        generated_seqs_for_target = decoded_sequences[start_idx:end_idx]

                        for seq_idx, generated_seq in enumerate(generated_seqs_for_target):
                            generated_seq = generated_seq.replace("C", "S")
                            # specially handle omit_AA, e.g., K and R
                            omit_AAs = set(self.global_config.models.proteinmpnn.omit_AAs)
                            if omit_AAs:
                                generated_seq = ''.join('N' if aa in omit_AAs else aa for aa in generated_seq)
                            unique_generated_id = f"binder_{int(time.time())}_{seq_idx}"
                            results.append({
                                'target_id': target_id,
                                'target_sequence': target_seq,  # Storing original full sequence
                                'input_target_region_sequence': input_region_seq,  # Save the actual input sequence
                                'binder_sequence': generated_seq,
                                'sampling_strategy': sampling_strategy,
                                'generation_params_str': params_str,
                                'generation_time_s': round(method_elapsed_time / len(target_ids_batch), 1),
                                'id': unique_generated_id,
                                'parent_id': unique_generated_id,
                                'original_parent_id': unique_generated_id
                            })

        overall_elapsed_time = time.time() - overall_start_time
        logger.info(
            f"Total generation for {len(targets_to_generate)} targets completed in {overall_elapsed_time:.2f} seconds.")

        return results