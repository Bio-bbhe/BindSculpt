import os
from Bio.PDB import PDBParser, ShrakeRupley
from pymol import cmd
from pathlib import Path


def find_surface_hydrophobic_residues(pdb_file, chain_id='B', threshold=25):
    # Define hydrophobic residues and max ASA values
    hydrophobic = {'GLY', 'ALA', 'CYS', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'TYR'}
    max_asa = {
        'GLY': 104.0, 'ALA': 129.0, 'VAL': 174.0, 'LEU': 201.0, 'ILE': 197.0, 'MET': 224.0, 'CYS': 167.0,
        'PHE': 240.0, 'TRP': 285.0, 'PRO': 159.0, 'TYR': 263.0,
    }

    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]  # Use first model

    # Fallback: Shrake-Rupley algorithm
    sr = ShrakeRupley()
    sr.compute(model, level="R")  # Compute residue-level SASA

    results = []
    hydrophobic_resi =[]
    full_pos = []
    # Process only specified chain
    if chain_id in model:
        chain = model[chain_id]
        k = 1
        for residue in chain:
            full_pos.append(k)
            k += 1
            # Skip non-amino acids
            hetatm_flag, res_num, icode = residue.id
            if hetatm_flag != " ":
                continue

            res_name = residue.get_resname()
            if res_name not in max_asa:
                continue

            sasa = residue.sasa  # Absolute SASA
            rsa = (sasa / max_asa[res_name]) * 100  # Convert to RSA %

            if res_name in hydrophobic and rsa >= threshold:
                results.append((chain_id, res_num, icode, res_name, rsa))
                if res_num:
                    hydrophobic_resi.append(res_num)
    else:
        print(f"Chain {chain_id} not found in structure")

    fixed_pos = [x for x in full_pos if x not in hydrophobic_resi]
    print(hydrophobic_resi)
    return fixed_pos


def count_residues_pymol(selection="all"):
    """Helper to count unique residues in a PyMOL selection."""
    model = cmd.get_model(selection)
    residues = set()
    for atom in model.atom:
        residues.add((atom.chain, atom.resi))  # Store as (chain, resi_str)
    # Convert resi to int for sorting, handle potential non-numeric resi if any
    contact_resn_int = sorted([int(r[1]) for r in residues if r[1].isdigit()])
    cmd.delete("all")
    return len(residues), contact_resn_int


def interface_resi_num(pdb_file_path: str, chain_to_design: str = "B",
                                         interacting_chain: str = "A", distance_threshold=3.5):
    """
    Uses PyMOL to identify interface residues on the chain_to_design
    that are interacting with interacting_chain.
    Returns a list of residue numbers (1-indexed) to be fixed.
    """
    print('running _get_interface_fixed_positions_pymol')
    try:
        protein_acc = Path(pdb_file_path).stem
        cmd.load(pdb_file_path, protein_acc)

        selection_design = f"chain {chain_to_design}"
        selection_target = f"chain {interacting_chain}"

        cmd.select(selection_design, f"chain {chain_to_design}")
        cmd.select(selection_target, f"chain {interacting_chain}")

        # Select residues in chain_to_design that are close to interacting_chain
        interaction_selection_name = f"interface_residues_on_{chain_to_design}"

        cmd.select(
            interaction_selection_name,
            f"byres ({selection_design} within {distance_threshold} of {selection_target})"
        )

        _, contact_resn_design_chain = count_residues_pymol(interaction_selection_name)

        cmd.delete("all")  # Clean up

        return contact_resn_design_chain
    except Exception as e:
        print(e)
        return "Counting num of interface residues failed"

# # Example usage:
# if __name__ == "__main__":
#     # pdb_file = '/home/hebeibei/Work/dl/code/BindSculpt/data/outputs_0627_8sm5_hs_0.5_t_0.1/af3_predictions/output_iteration_15_batch_1751050167/binder_1751040249_6__175104816712/binder_1751040249_6__175104816712_model.pdb'
#     # pdb_file = 'data/outputs_v0628_3bik_PD1/af3_predictions/output_iteration_11_batch_1751126585/binder_1751120077_0__1751125535682/binder_1751120077_0__1751125535682_model.pdb'
#     pdb_file = '/home/hebeibei/Work/dl/code/BindSculpt/data/outputs_v0704_9J8Z_template/run_001_temp2.0/af3_predictions/output_iteration_0_batch_1751623305/binder_1751623305_9/binder_1751623305_9_model.pdb'
#     # Analyze chain B with 25% RSA threshold
#     surface_hydrophobic = find_surface_hydrophobic_residues(
#         pdb_file,
#         chain_id='B',
#         threshold=25
#     )
#     print(surface_hydrophobic)
#
#     interface = interface_resi_num(pdb_file,"A","B")
#     print(interface)
    # # Print results
    # print(f"Surface-exposed hydrophobic residues in chain B (ResNum, ResName, RSA%):")
    # for res in surface_hydrophobic:
    #     # Handle insertion codes
    #     res_id = f"{res[1]}{res[2]}" if res[2] != " " else res[1]
    #     print(f"Chain {res[0]}, Residue {res_id} ({res[3]}), RSA: {res[4]:.2f}%")
