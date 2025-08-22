# NOTE: This function is a refactored and modified version of the original
# `ipsae.py` script developed by Roland Dunbrack. It preserves the core
# scientific logic of the original implementation.
#
# Original source repository:
# https://github.com/DunbrackLab/IPSAE
#
# Key modifications include:
# - Encapsulation of the script's logic into a single Python function.
# - Removal of all file I/O operations (no .txt or .pml files are created).
# - Direct return of the calculated ipSAE_min and LIS scores.

# === 1. IMPORTS AND HELPER FUNCTIONS ===
# These are the original imports and helper functions, moved outside the main function
# for better organization and to avoid re-defining them on every function call.

import sys, os, math
import json
import numpy as np

np.set_printoptions(threshold=np.inf)  # Maintained from original for potential debugging


# The ptm_func and d0 calculation functions are unchanged from the original script.
# They are fundamental to the ipSAE calculation.
def ptm_func(x, d0):
    return 1.0 / (1 + (x / d0) ** 2.0)


ptm_func_vec = np.vectorize(ptm_func)


def calc_d0(L, pair_type):
    L = float(L)
    if L < 27: L = 27
    min_value = 1.0
    if pair_type == 'nucleic_acid': min_value = 2.0
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    return max(min_value, d0)


def calc_d0_array(L, pair_type):
    L = np.array(L, dtype=float)
    L = np.maximum(27, L)
    min_value = 1.0
    if pair_type == 'nucleic_acid': min_value = 2.0
    return np.maximum(min_value, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)


# The PDB and CIF parsing functions are also unchanged. They are required to
# extract atomic coordinates, chain IDs, and residue information from the structure file.
def parse_pdb_atom_line(line):
    atom_num = int(line[6:11].strip())
    atom_name = line[12:16].strip()
    residue_name = line[17:20].strip()
    chain_id = line[21].strip()
    residue_seq_num = int(line[22:26].strip())
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    return {'atom_num': atom_num, 'atom_name': atom_name, 'residue_name': residue_name,
            'chain_id': chain_id, 'residue_seq_num': residue_seq_num, 'x': x, 'y': y, 'z': z}


def parse_cif_atom_line(line, fielddict):
    linelist = line.split()
    residue_seq_num_str = linelist[fielddict['label_seq_id']]
    if residue_seq_num_str == ".": return None
    atom_num = int(linelist[fielddict['id']])
    atom_name = linelist[fielddict['label_atom_id']]
    residue_name = linelist[fielddict['label_comp_id']]
    chain_id = linelist[fielddict['label_asym_id']]
    residue_seq_num = int(residue_seq_num_str)
    x = float(linelist[fielddict['Cartn_x']])
    y = float(linelist[fielddict['Cartn_y']])
    z = float(linelist[fielddict['Cartn_z']])
    return {'atom_num': atom_num, 'atom_name': atom_name, 'residue_name': residue_name,
            'chain_id': chain_id, 'residue_seq_num': residue_seq_num, 'x': x, 'y': y, 'z': z}


# The dictionary initialization helpers are also preserved from the original.
def init_chainpairdict_zeros(chainlist):
    return {chain1: {chain2: 0 for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}


def init_chainpairdict_npzeros(chainlist, arraysize):
    return {chain1: {chain2: np.zeros(arraysize) for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}


# The chain classification helper is preserved.
def classify_chains(chains, residue_types):
    nuc_residue_set = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
    chain_types = {}
    unique_chains = np.unique(chains)
    for chain in unique_chains:
        indices = np.where(chains == chain)[0]
        chain_residues = residue_types[indices]
        nuc_count = sum(residue in nuc_residue_set for residue in chain_residues)
        chain_types[chain] = 'nucleic_acid' if nuc_count > 0 else 'protein'
    return chain_types


# === 2. MAIN FUNCTION DEFINITION ===
# The entire script is now encapsulated in this function.
def calculate_ipsae_min(pae_file_path, pdb_cif_path, target_chain_id, binder_chain_id, pae_cutoff=10.0):
    """
    Calculates the ipSAE_min score for a target-binder pair from AlphaFold/Boltz1 outputs.

    This function is a refactoring of the original ipsae.py script. It performs all calculations
    in memory and directly returns the ipSAE_min value without writing any files.

    Args:
        pae_file_path (str): Path to the PAE data file (.json for AF2/AF3, .npz for Boltz1).
        pdb_cif_path (str): Path to the corresponding structure file (.pdb for AF2, .cif for AF3/Boltz1).
        target_chain_id (str): The chain identifier for the target protein (e.g., 'A').
        binder_chain_id (str): The chain identifier for the binder protein (e.g., 'B').
        pae_cutoff (float, optional): The PAE cutoff to define interacting residues. Defaults to 10.0.

    Returns:
        float: The calculated ipSAE_min score, or None if chains are not found.
    """

    # === 3. FILE TYPE AND MODEL DETECTION ===
    # This logic is adapted from the original script to determine the model (AF2, AF3, Boltz1)
    # based on the file extensions of the input paths.
    if ".pdb" in pdb_cif_path:
        cif = False
    elif ".cif" in pdb_cif_path and pae_file_path.endswith(".json"):
        cif = True
    elif ".cif" in pdb_cif_path and pae_file_path.endswith(".npz"):
        cif = True
    else:
        print(f"Error: Unsupported PDB/CIF file type: {pdb_cif_path}")
        return None

    # === 4. STRUCTURE FILE PARSING ===
    # This section reads the PDB or CIF file to extract coordinates and chain info.
    # It remains identical to the original script's logic.
    residues = []
    chains = []
    token_mask = []
    residue_set = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                   "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
                   "DA", "DC", "DT", "DG", "A", "C", "U", "G"}

    atomsitefield_num = 0
    atomsitefield_dict = {}
    try:
        with open(pdb_cif_path, 'r') as PDB:
            for line in PDB:
                if line.startswith("_atom_site."):
                    line = line.strip()
                    (atomsite, fieldname) = line.split(".")
                    atomsitefield_dict[fieldname] = atomsitefield_num
                    atomsitefield_num += 1

                if line.startswith("ATOM") or line.startswith("HETATM"):
                    if cif:
                        atom = parse_cif_atom_line(line, atomsitefield_dict)
                    else:
                        atom = parse_pdb_atom_line(line)
                    if atom is None:
                        token_mask.append(0)
                        continue
                    if atom['atom_name'] == "CA" or "C1" in atom['atom_name']:
                        token_mask.append(1)
                        # This now correctly adds the residue name to the dictionary
                        residues.append({
                            'coor': np.array([atom['x'], atom['y'], atom['z']]),
                            'res': atom['residue_name']
                        })
                        chains.append(atom['chain_id'])
                    if atom['atom_name'] != "CA" and "C1" not in atom['atom_name'] and atom[
                        'residue_name'] not in residue_set:
                        token_mask.append(0)
    except FileNotFoundError:
        print(f"Error: Structure file not found at {pdb_cif_path}")
        return None

    # === 5. DATA PREPARATION AND VALIDATION ===
    # Converts lists to NumPy arrays for efficient computation.
    numres = len(residues)
    if numres == 0:
        print(f"Error: No residues found in {pdb_cif_path}")
        return None

    chains = np.array(chains)
    unique_chains = np.unique(chains)
    token_array = np.array(token_mask)
    residue_types = np.array(
        [res['res'] for res in residues] if 'res' in residues[0] else [])  # Simplified for this function

    # MODIFICATION: Check if the user-provided chain IDs exist in the structure.
    if target_chain_id not in unique_chains or binder_chain_id not in unique_chains:
        print(f"Error: Target chain '{target_chain_id}' or Binder chain '{binder_chain_id}' not found.")
        print(f"Available chains: {unique_chains}")
        return None

    # This logic for determining chain types (protein/nucleic acid) is preserved.
    chain_dict = classify_chains(chains, residue_types)
    chain_pair_type = init_chainpairdict_zeros(unique_chains)

    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2: continue
            if chain_dict.get(c1) == 'nucleic_acid' or chain_dict.get(c2) == 'nucleic_acid':
                chain_pair_type[c1][c2] = 'nucleic_acid'
            else:
                chain_pair_type[c1][c2] = 'protein'

    # === 6. PAE MATRIX LOADING ===
    # This extensive logic block loads the PAE matrix depending on the model type.
    # It is preserved exactly to handle AF2, AF3, and Boltz1 formats correctly.
    pae_matrix = None
    try:
        if ".pdb" in pdb_cif_path:  # AF2
            if pae_file_path.endswith('.pkl'):
                data = np.load(pae_file_path, allow_pickle=True)
            else:
                with open(pae_file_path, 'r') as file:
                    data = json.load(file)
            if 'pae' in data:
                pae_matrix = np.array(data['pae'])
            elif 'predicted_aligned_error' in data:
                pae_matrix = np.array(data['predicted_aligned_error'])
        elif ".cif" in pdb_cif_path and pae_file_path.endswith(".json"):  # AF3
            with open(pae_file_path, 'r') as file:
                data = json.load(file)
            if 'pae' in data:
                pae_matrix_af3 = np.array(data['pae'])
            else:
                raise KeyError("No PAE data in AF3 json file")
            pae_matrix = pae_matrix_af3[np.ix_(token_array.astype(bool), token_array.astype(bool))]
        elif ".cif" in pdb_cif_path and pae_file_path.endswith(".npz"):  # Boltz1
            data_pae = np.load(pae_file_path)
            pae_matrix_boltz1 = np.array(data_pae['pae'])
            pae_matrix = pae_matrix_boltz1[np.ix_(token_array.astype(bool), token_array.astype(bool))]
    except FileNotFoundError:
        print(f"Error: PAE file not found at {pae_file_path}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error reading PAE file {pae_file_path}: {e}")
        return None

    if pae_matrix is None:
        print(f"Error: Could not load PAE matrix from {pae_file_path}")
        return None

    # === 7. SCORE CALCULATION ===
    # This is the core calculation loop. It is functionally identical to the original script.
    # Dictionaries are initialized to store intermediate and final scores.
    ipsae_d0res_byres = init_chainpairdict_npzeros(unique_chains, numres)
    ipsae_d0res_asym = init_chainpairdict_zeros(unique_chains)
    n0res_byres = init_chainpairdict_npzeros(unique_chains, numres)
    d0res_byres = init_chainpairdict_npzeros(unique_chains, numres)
    LIS = init_chainpairdict_zeros(unique_chains)

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2: continue

            valid_pairs_matrix = (chains == chain2) & (pae_matrix < pae_cutoff)
            n0res_byres_all = np.sum(valid_pairs_matrix, axis=1)
            d0res_byres_all = calc_d0_array(n0res_byres_all, chain_pair_type.get(chain1, {}).get(chain2, 'protein'))

            n0res_byres[chain1][chain2] = n0res_byres_all
            d0res_byres[chain1][chain2] = d0res_byres_all

            # Create a boolean mask for the specific chain pair
            mask = (chains[:, None] == chain1) & (chains[None, :] == chain2)
            selected_pae = pae_matrix[mask]

            if selected_pae.size > 0:
                # Filter for PAE values under the LIS-specific cutoff of 12
                valid_pae = selected_pae[selected_pae <= 12]
                if valid_pae.size > 0:
                    # Calculate and store the mean score
                    scores = (12 - valid_pae) / 12
                    LIS[chain1][chain2] = np.mean(scores)
                else:
                    LIS[chain1][chain2] = 0  # Score is 0 if no residues meet the cutoff
            else:
                LIS[chain1][chain2] = 0

            for i in range(numres):
                if chains[i] != chain1: continue
                valid_pairs = valid_pairs_matrix[i]
                ptm_row_d0res = ptm_func_vec(pae_matrix[i], d0res_byres[chain1][chain2][i])
                ipsae_d0res_byres[chain1][chain2][i] = ptm_row_d0res[valid_pairs].mean() if valid_pairs.any() else 0.0

    # This loop computes the final asymmetric scores by finding the max score from each residue's perspective.
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2: continue
            interchain_values = ipsae_d0res_byres[chain1][chain2]
            # The original script finds the max over the residues of the first chain.
            max_index = np.argmax(interchain_values)
            ipsae_d0res_asym[chain1][chain2] = interchain_values[max_index]

    # === 8. EXTRACT AND RETURN IPSAE_MIN ===
    # MODIFICATION: Instead of writing all pairs to a file, we now extract the specific
    # scores for the requested target-binder pair and calculate the minimum.
    try:
        score_target_to_binder = ipsae_d0res_asym[target_chain_id][binder_chain_id]
        score_binder_to_target = ipsae_d0res_asym[binder_chain_id][target_chain_id]

        lis_t_to_b = LIS[target_chain_id][binder_chain_id]
        lis_b_to_t = LIS[binder_chain_id][target_chain_id]

    except KeyError:
        print("Error: A key error occurred while trying to access calculated scores.")
        print("This may happen if one of the chains has no valid interacting residues.")
        return None

    # The final ipSAE_min is the minimum of the two asymmetric scores.
    ipsae_min = min(score_target_to_binder, score_binder_to_target)
    lis_score = min(lis_t_to_b,lis_b_to_t)

    return round(ipsae_min,3),round(lis_score,3)


# === 9. EXAMPLE USAGE ===
# This demonstrates how to use the new function.
# if __name__ == '__main__':
    # # You would replace these placeholder paths with your actual file paths.
    # # Example for AlphaFold3:
    # pae_file = '/home/hebeibei/Work/dl/code/BindSculpt/tests/Latent_labs/fold_ll_mini_bhrf1_13/fold_ll_mini_bhrf1_13_full_data_0.json'
    # cif_file = '/home/hebeibei/Work/dl/code/BindSculpt/tests/Latent_labs/fold_ll_mini_bhrf1_13/fold_ll_mini_bhrf1_13_model_0.cif'
    # target = 'A'
    # binder = 'B'
    #
    # # Check if placeholder files exist before running
    # if os.path.exists(pae_file) and os.path.exists(cif_file):
    #     # Call the function with your specific chain IDs
    #     final_ipsae_min,lis_score = calculate_ipsae_min(pae_file, cif_file, target, binder)
    #
    #     if final_ipsae_min is not None:
    #         print(f"\nCalculation successful!")
    #         print(f"Target Chain: {target}")
    #         print(f"Binder Chain: {binder}")
    #         print(f"Final ipSAE_min Score: {final_ipsae_min},{lis_score}")
    # else:
    #     print("Example usage skipped: Please update placeholder file paths in the `if __name__ == '__main__':` block.")
