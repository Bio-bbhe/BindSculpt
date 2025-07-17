# src/utils/helpers.py
import yaml
import logging
import os
from omegaconf import OmegaConf  # if using, otherwise simple yaml.safe_load
from Bio import Align  # Add this import for compute_identity
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path="config/config.yaml"):
    """
    Loads the YAML config and merges it with command-line overrides.
    """
    try:
        # Get absolute path relative to project root
        if os.environ.get('PROJECT_ROOT'):
            config_path = os.path.join(os.environ['PROJECT_ROOT'], config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        # Load and resolve config
        # Load the base config from the file
        base_conf = OmegaConf.load(config_path)

        # Load command-line arguments as a config object
        cli_conf = OmegaConf.from_cli()

        # Merge the two, with command-line arguments taking precedence
        conf = OmegaConf.merge(base_conf, cli_conf)

        # Resolve any variables (like ${paths.data_outputs})
        OmegaConf.resolve(conf)

        logger.info("Configuration loaded and merged with command-line overrides.")
        return conf

    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def setup_logging(log_level="INFO", log_dir="logs", log_file="bindsculpt.log"):
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_file)

    # Basic logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()  # Output to console as well
        ]
    )
    logger.info("Logging setup complete.")


def compute_identity(true_seq, pred_seq):
    """
    Computes sequence identity, coverage, and adjusted identity.
    Moved from original generator script.
    """
    aligner = Align.PairwiseAligner()
    alignments = aligner.align(true_seq, pred_seq)
    if not alignments:
        return 0.0, 0.0, 0.0, 0.0  # No alignment found

    aligned_seq1, aligned_seq2 = alignments[0]

    matches = len([i for i, k in zip(aligned_seq1, aligned_seq2) if i == k and i != '-'])

    # Handle cases where min(len(true_seq), len(pred_seq)) could be zero
    min_len = min(len(true_seq), len(pred_seq))
    identity = (matches / min_len * 100) if min_len > 0 else 0.0

    # Ensure coordinates exist and are valid for coverage calculation
    if alignments[0].coordinates.shape[1] < 2:  # Check if there are at least two columns (start and end)
        true_seq_coverage = 0.0
        pred_seq_coverage = 0.0
    else:
        true_seq_coverage = (alignments[0].coordinates[0][-1] - alignments[0].coordinates[0][0]) / len(true_seq) if len(
            true_seq) > 0 else 0.0
        pred_seq_coverage = (alignments[0].coordinates[1][-1] - alignments[0].coordinates[1][0]) / len(pred_seq) if len(
            pred_seq) > 0 else 0.0

    # Ensure coverages are between 0 and 1, inclusive
    true_seq_coverage = max(0.0, min(1.0, true_seq_coverage))
    pred_seq_coverage = max(0.0, min(1.0, pred_seq_coverage))

    w = 0.5  # ! here w means how much weights are focused on true seq coverage
    adjusted_identity = identity * (w * (true_seq_coverage) + (1 - w) * (pred_seq_coverage))

    return round(identity, 1), round(true_seq_coverage, 2), round(pred_seq_coverage, 2), round(adjusted_identity, 2)


def get_bad_word_ids_for_tokenizer(tokenizer):
    """
    Generates a list of bad word IDs from a hardcoded token map.
    It's more robust to define valid amino acids and infer bad words.
    """
    # This token map should ideally be derived from the tokenizer's vocabulary
    # or passed as an argument if it's dynamic. For now, using the hardcoded one.
    # token_map = {
    #     '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7,
    #     'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16,
    #     'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25,
    #     'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32
    # }
    #
    # valid_tokens = ['<cls>', '<pad>', '<eos>',
    #                 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q',
    #                 'N', 'F', 'Y', 'M', 'H', 'W', 'C']
    #
    # valid_token_ids = {token_map[token] for token in valid_tokens if token in token_map}
    # bad_token_ids = [[token_map[token]] for token in token_map if token_map[token] not in valid_token_ids]

    # As a more robust alternative, if the tokenizer uses standard amino acid tokens
    # and you want to exclude non-standard ones, you can do:
    standard_aas = set("ACDEFGHIKLMNPQRSTVWY")
    bad_token_ids = []
    for token, token_id in tokenizer.get_vocab().items():
        if token.upper() not in standard_aas:  # len(token) == 1 and token.isalpha() and
            bad_token_ids.append([token_id])
    # Add special tokens if they are not supposed to appear in generated seqs, e.g., <unk>, <mask>
    if tokenizer.unk_token_id is not None:
        bad_token_ids.append([tokenizer.unk_token_id])
    if tokenizer.mask_token_id is not None:
        bad_token_ids.append([tokenizer.mask_token_id])

    return bad_token_ids


# check if generated binder is indeed in the defined binding_region in the predicted complex
def validate_binder_contacts(
        contact_residues: list[int],
        binding_region_str: str,
        threshold: float = 0.8
):
    """
    Validates if a binder's contacts on a target fall within a defined binding region.

    Args:
        contact_residues (list[int]): A list of 1-based residue numbers on the target
                                      that are in contact with the binder.
                                      Example: [1, 2, 3, 5, 154, 155]
        binding_region_str (str): A string defining the allowed binding regions on the target.
                                  Example: "1-11,68-84,148-170,241-253"
        threshold (float): The minimum required fraction (0.0 to 1.0) of contact residues
                           that must be within the defined region for the binder to be
                           considered qualified. Defaults to 1.0 (strict 100% overlap).

    Returns:
        dict: A dictionary containing the validation results:
              - 'is_qualified' (bool): True if the overlap meets the threshold.
              - 'overlap_percentage' (float): The percentage of contacts within the region.
              - 'inside_region_contacts' (list[int]): Contacts that are in the region.
              - 'outside_region_contacts' (list[int]): Contacts that are outside the region.
              - 'total_contacts' (int): Total number of contacts.
    """

    # Helper to parse the region string into a set of 1-based integer indices
    def _parse_region_to_set(region_str: str) -> set[int]:
        allowed_indices = set()
        if not region_str:
            return allowed_indices
        parts = region_str.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    allowed_indices.update(range(start, end + 1))
                except ValueError:
                    print(f"Warning: Could not parse range '{part}'. Skipping.")
            else:
                try:
                    allowed_indices.add(int(part))
                except ValueError:
                    print(f"Warning: Could not parse index '{part}'. Skipping.")
        return allowed_indices

    # 1. Parse the allowed binding region into a set for efficient lookup
    allowed_residues_set = _parse_region_to_set(binding_region_str)

    # Handle edge case of no contacts
    if not contact_residues:
        is_in_defined_binding_region = False
        return is_in_defined_binding_region

    # 2. Find which contacts are inside and outside the allowed region
    actual_contact_set = set(contact_residues)
    inside_contacts = sorted(list(actual_contact_set.intersection(allowed_residues_set)))

    # 3. Calculate the overlap percentage
    total_contacts = len(actual_contact_set)
    overlap_percentage = len(inside_contacts) / total_contacts if total_contacts > 0 else 0.0

    # 4. Determine if it's qualified based on the threshold
    is_in_defined_binding_region = (overlap_percentage >= threshold)

    return is_in_defined_binding_region
