# src/filters/rules.py
import logging

logger = logging.getLogger(__name__)


class BinderFilters:
    def __init__(self, config):
        self.config = config.filters
        logger.info("Binder filters initialized.")

    def apply_sequence_filters(self, binder_data):
        """
        Applies sequence-based filters to a single binder dictionary.

        Args:
            binder_data (dict): A dictionary containing binder sequence and metadata.

        Returns:
            tuple: (bool, str) - (True if passed, False otherwise), and a reason string.
        """
        if not self.config.sequence_filters.enabled:
            return True, "Sequence filters disabled"

        binder_seq = binder_data.get('binder_sequence', '')
        if not binder_seq:
            return False, "Missing binder sequence"

        # Length filter
        min_len = self.config.sequence_filters.min_length
        max_len = self.config.sequence_filters.max_length
        if not (min_len <= len(binder_seq) <= max_len):
            return False, f"Length ({len(binder_seq)}) outside [{min_len}, {max_len}]"

        # Prohibited motifs filter
        max_cys_num = self.config.sequence_filters.max_cys_num
        cys_num = binder_seq.count('C')
        if max_cys_num < cys_num:
            return False, f"Contains {cys_num} Cys residues"

        # Add more sequence filters here (e.g., hydrophobicity, charge, specific amino acid counts)

        return True, "Passed sequence filters"

    def apply_complex_filters(self, binder_data):
        """
        Applies complex-based filters to a single binder dictionary using AF3 scores.

        Args:
            binder_data (dict): A dictionary containing binder sequence and AF3 scores.

        Returns:
            tuple: (bool, str) - (True if passed, False otherwise), and a reason string.
        """
        if not self.config.complex_filters.enabled:
            return True, "Complex filters disabled"

        iptm = binder_data.get('iptm')
        ptm = binder_data.get('ptm')

        if iptm is None or ptm is None:
            # This might happen if AF3 prediction failed or was skipped
            return False, "Missing AF3 scores for complex filtering"

        # IPTM filter
        min_iptm = self.config.complex_filters.min_iptm
        if iptm < min_iptm:
            return False, f"IPTM ({iptm:.2f}) below threshold ({min_iptm})"

        # pLDDT filter
        min_ptm = self.config.complex_filters.min_ptm
        if ptm < min_ptm:
            return False, f"pLDDT ({ptm:.2f}) below threshold ({min_ptm})"

        # Add more complex filters here (e.g., based on interface residues, predicted clashes)

        return True, "Passed complex filters"

    def apply_all_filters(self, binder_data):
        """
        Applies all enabled filters in sequence.
        """
        passed_seq, reason_seq = self.apply_sequence_filters(binder_data)
        if not passed_seq:
            logger.debug(f"Binder failed sequence filter: {reason_seq}")
            return False, reason_seq

        passed_comp, reason_comp = self.apply_complex_filters(binder_data)
        if not passed_comp:
            logger.debug(f"Binder failed complex filter: {reason_comp}")
            return False, reason_comp

        return True, "Passed all filters"