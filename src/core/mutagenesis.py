# src/core/mutagenesis.py (Full code)
import logging
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import random  # For random mutations
import math
import uuid
import time

logger = logging.getLogger(__name__)


class Mutagenesis:
    def __init__(self, config):
        self.config = config.mutagenesis
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"  # Standard amino acids
        logger.info("Mutagenesis module initialized.")

    def generate_single_mutants(self, parent_sequence: str, parent_id: str, original_parent_id: str, iteration_num: int):
        """
        Generates all single amino acid mutants for a given parent sequence.
        """
        mutant_pool = []
        logger.debug(f"Generating single mutants for parent {parent_id} (Length: {len(parent_sequence)}).")

        k = 0
        for i in range(len(parent_sequence)):
            original_aa = parent_sequence[i]
            for mutant_aa in self.aa_alphabet:
                if mutant_aa != original_aa:
                    k += 1
                    mutant_seq_list = list(parent_sequence)
                    mutant_seq_list[i] = mutant_aa
                    mutant_seq = "".join(mutant_seq_list)
                    # mutant_id = f"{parent_id}_S_{original_aa}{i + 1}{mutant_aa}"  # Added _S for single mutant
                    # mutant_id = f"{parent_id.split('__')[0]}__{int(time.time())}_i{iteration_num}_{k}"
                    mutant_id = f"{parent_id.split('__')[0]}__{uuid.uuid4().hex[:16]}"

                    mutant_pool.append({
                        'id': mutant_id,
                        'binder_sequence': mutant_seq,
                        'parent_id': parent_id,  # Link back to parent
                        'original_parent_id': original_parent_id,
                        'mutation_type': 'single_site',
                        'original_aa_pos_mut': f"{original_aa}{i + 1}{mutant_aa}",  # e.g., A10B
                        'af3_status': 'Not processed yet'
                    })
        logger.info(f"Generated {len(mutant_pool)} single mutants for parent {original_parent_id}.")
        return mutant_pool

    def generate_single_plus_one_mutants(self, parent_sequence: str, parent_id: str, original_parent_id: str,
                                         iteration_num: int):
        """
        Generates all single amino acid mutants PLUS one additional random mutation
        for each single mutant (creating double mutants).
        """
        mutant_pool = []
        logger.debug(f"Generating single+1 mutants for parent {parent_id} (Length: {len(parent_sequence)}).")

        k = 0
        seq_length = len(parent_sequence)

        for i in range(seq_length):
            original_aa = parent_sequence[i]

            for mutant_aa in self.aa_alphabet:
                if mutant_aa != original_aa:
                    # Generate single mutant
                    k += 1
                    mutant_seq_list = list(parent_sequence)
                    mutant_seq_list[i] = mutant_aa
                    single_mutant_seq = "".join(mutant_seq_list)
                    single_mutant_id = f"{parent_id.split('__')[0]}__{int(time.time())}{k}_S"

                    # Add single mutant to pool
                    mutant_pool.append({
                        'id': single_mutant_id,
                        'binder_sequence': single_mutant_seq,
                        'parent_id': parent_id,
                        'original_parent_id': original_parent_id,
                        'mutation_type': 'single_site',
                        'original_aa_pos_mut': f"{original_aa}{i + 1}{mutant_aa}",
                        'af3_status': 'Not processed yet'
                    })

                    # Generate double mutant: single mutation + one additional random mutation
                    # Choose a different position that hasn't been mutated
                    valid_positions = [pos for pos in range(seq_length) if pos != i]
                    if not valid_positions:  # Skip if no valid positions
                        continue

                    j = random.choice(valid_positions)
                    original_aa_j = parent_sequence[j]
                    valid_aas = [aa for aa in self.aa_alphabet if aa != original_aa_j]

                    if valid_aas:
                        k += 1
                        mutant_aa_j = random.choice(valid_aas)
                        double_mutant_seq_list = list(single_mutant_seq)
                        double_mutant_seq_list[j] = mutant_aa_j
                        double_mutant_seq = "".join(double_mutant_seq_list)
                        double_mutant_id = f"{parent_id.split('__')[0]}__{uuid.uuid4().hex[:16]}"

                        # Add double mutant to pool
                        mutant_pool.append({
                            'id': double_mutant_id,
                            'binder_sequence': double_mutant_seq,
                            'parent_id': single_mutant_id,  # Parent is the single mutant
                            'original_parent_id': original_parent_id,
                            'mutation_type': 'double_site',
                            'original_aa_pos_mut': f"{original_aa}{i + 1}{mutant_aa},{original_aa_j}{j + 1}{mutant_aa_j}",
                            'af3_status': 'Not processed yet'
                        })

        logger.info(f"Generated {len(mutant_pool)} mutants ({k} mutations) for parent {original_parent_id}.")
        return mutant_pool

    def generate_random_multiple_mutants(self, parent_sequence: str, parent_id: str, original_parent_id: str, iteration_num:int):
        """
        Generates a specified number of mutants with random mutations up to max_mutations.
        Parameters come from config.
        """
        target_mut_count = math.ceil(len(parent_sequence) * self.config.multiple_site.mutation_rate_percent)
        min_muts = self.config.multiple_site.min_mutations_per_binder
        max_muts = self.config.multiple_site.max_mutations_per_binder
        num_actual_mutations = max(min_muts, min(target_mut_count, max_muts))
        num_mutants_to_generate = self.config.multiple_site.num_mutants_per_parent

        mutant_pool = []
        seq_len = len(parent_sequence)

        if seq_len == 0:
            logger.warning(f"Parent sequence for {parent_id} is empty. Cannot generate mutants.")
            return []

        logger.debug(
            f"Generating {num_mutants_to_generate} random multiple mutants (max {max_muts} mutations) for parent {parent_id}.")

        for k in range(num_mutants_to_generate):
            mutated_seq_list = list(parent_sequence)

            # Ensure unique positions for mutation
            mutation_positions = random.sample(range(seq_len), min(num_actual_mutations, seq_len))
            mutation_positions.sort()  # Sort to ensure consistent ID naming

            mutations_made_str = []
            for pos in mutation_positions:
                original_aa = mutated_seq_list[pos]
                possible_mutants = [aa for aa in self.aa_alphabet if aa != original_aa]
                if possible_mutants:
                    new_aa = random.choice(possible_mutants)
                    mutated_seq_list[pos] = new_aa
                    mutations_made_str.append(f"{original_aa}{pos + 1}{new_aa}")

            mutant_seq = "".join(mutated_seq_list)

            # mutant_id = f"{parent_id}_M_{'_'.join(mutations_made_str)}"  # _M for multiple mutant
            mutant_id = f"{parent_id.split('__')[0]}__{uuid.uuid4().hex[:16]}"

            if len(mutations_made_str) == 0:  # Handle case where no actual mutations could be made (e.g., all positions already ideal)
                mutant_id = f"{parent_id}_M"  # Add a specific tag

            mutant_pool.append({
                'id': mutant_id,
                'binder_sequence': mutant_seq,
                'parent_id': parent_id,
                'original_parent_id': original_parent_id,
                'mutation_type': 'multiple_site',
                'mutations_applied': mutations_made_str,  # Store as list of strings
                'af3_status': 'Not processed yet'
            })
        logger.info(f"Generated {len(mutant_pool)} random multiple mutants for parent {original_parent_id}.")
        return mutant_pool