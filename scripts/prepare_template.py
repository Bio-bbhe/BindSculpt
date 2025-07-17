from Bio.PDB import MMCIFParser, MMCIFIO
import warnings
import argparse # Import argparse

def parser_cif(cif_file, output_file, target_chain_id, structure_id='template'):
    """
    Parses a CIF file, filters it to keep only the target chain,
    removes water molecules, and saves the result.

    Args:
        cif_file (str): Path to the input CIF file.
        output_file (str): Path to save the filtered CIF file.
        target_chain_id (str): The ID of the chain to keep.
        structure_id (str, optional): ID for the structure. Defaults to 'template'.

    Returns:
        str: The path to the saved output file.
    """
    parser = MMCIFParser(QUIET=True)
    with warnings.catch_warnings(): # Suppress PDBConstructionWarning
        warnings.simplefilter("ignore")
        struct = parser.get_structure(structure_id, cif_file)

    for model in struct:
        # First pass: remove water molecules (residues starting with 'W')
        for chain in model:
            # Create a list of water residue IDs to avoid issues with modifying while iterating
            water_residue_ids = [res.id for res in list(chain) if res.id[0] == "W"]
            for wid in water_residue_ids:
                chain.detach_child(wid)

        # Second pass: remove empty chains and chains not matching target_chain_id
        # Create a list of chain IDs to avoid issues with modifying while iterating
        chain_ids_to_process = [chain.id for chain in list(model)]
        for chain_id in chain_ids_to_process:
            chain = model[chain_id] # Get the chain object again as it might have been modified
            if len(chain) == 0:
                model.detach_child(chain_id)
            elif chain_id != target_chain_id:
                model.detach_child(chain_id)

    io = MMCIFIO()
    io.set_structure(struct)
    io.save(output_file)
    return output_file


def insert_lines_after(filepath, target_line, lines_to_insert):
    """
    Inserts a list of lines after the first occurrence of a target line in a file.

    Args:
        filepath (str): The path to the file to modify.
        target_line (str): The line (or part of it) after which to insert.
        lines_to_insert (list): A list of strings to insert, each as a new line.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    inserted = False # Flag to ensure insertion happens only once
    for line in lines:
        new_lines.append(line)
        if not inserted and target_line in line:
            for insert_line in lines_to_insert:
                new_lines.append(insert_line + '\n')
            inserted = True # Mark as inserted

    with open(filepath, 'w') as f:
        f.writelines(new_lines)


if __name__ == '__main__':
    # --- Argument Parsing ---
    arg_parser = argparse.ArgumentParser(
        description="Filters a CIF file to a specific chain and adds header information."
    )
    arg_parser.add_argument("--input_file",type=str,default='/home/hebeibei/Work/dl/code/BindSculpt/tests/3bik.cif',
        help="Path to the input CIF file (e.g., /path/to/3bik.cif)")
    arg_parser.add_argument("--output_file",type=str,default='/home/hebeibei/Work/dl/code/BindSculpt/tests/template_filtered.cif',
        help="Path to save the filtered output CIF file (e.g., /path/to/template_filtered.cif)")
    arg_parser.add_argument("--target_chain_id", type=str,default='A', help="The ID of the chain to keep (e.g., B)")
    arg_parser.add_argument("--structure_id",type=str,default="template", help="ID for the PDB structure (default: template)")
    args = arg_parser.parse_args()

    # Use parsed arguments
    input_cif = args.input_file
    output_cif = args.output_file
    chain_id = args.target_chain_id
    struct_id = args.structure_id

    print(f"Processing CIF file: {input_cif}")
    print(f"Target chain ID: {chain_id}")
    print(f"Output file: {output_cif}")
    print(f"Structure ID: {struct_id}")

    # Call the parser function
    file_path = parser_cif(input_cif, output_cif, target_chain_id=chain_id, structure_id=struct_id)
    print(f"Filtered CIF saved to: {file_path}")

    # Line to match for inserting header info (can be partial or full line)
    target_header_line = f'data_{struct_id}' # Use the parsed structure_id

    # Lines to insert after the target line
    lines_to_insert = [
        f'_entry.id                        {struct_id}',
        '#',
        '_pdbx_audit_revision_history.revision_date 2005-12-27', # Example date, can be parameterized if needed
        '#',
        '_pdbx_database_status.recvd_initial_deposition_date 2005-12-27' # Example date
    ]

    insert_lines_after(file_path, target_header_line, lines_to_insert)
    print(f"Header lines inserted into: {file_path}")
    print("Processing complete.")