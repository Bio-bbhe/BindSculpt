# src/models/structural_predictor.py
import logging
import os
import subprocess
import json
import time
import pyrosetta as pr
from pathlib import Path
from tqdm import tqdm  # For progress bar
from pymol import cmd as pymol_cmd
from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
from src.utils.structure_utils import interface_resi_num
from src.utils.helpers import validate_binder_contacts


logger = logging.getLogger(__name__)


class AlphaFold3Predictor:
    def __init__(self, config):
        self.config = config.models.alphafold3
        self.global_config = config  # Keep a reference to global config for paths
        self.executable_path = self.config.executable_path

        self.db_dir = self.config.db_dir
        self.model_dir = self.config.model_dir

        self.target_msa_dir = config.paths.target_msa_dir
        self.target_cif_path = config.paths.target_cif_path
        self.conda_env_name = self.config.conda_env_name
        self.cpu = self.config.cpu
        self.gpu = self.config.CUDA_VISIBLE_DEVICES

        # Base directory within the project for ALL AF3 runs' outputs (each run gets a subdirectory)
        self.project_af3_output_base_dir = os.path.join(config.paths.run_output_dir, self.config.output_subdir)
        os.makedirs(self.project_af3_output_base_dir, exist_ok=True)

        # Base directory for temporary AF3 input directories (each batch run gets its own temp input dir)
        self.af3_input_temp_base_dir = os.path.join(self.project_af3_output_base_dir, "af3_batch_inputs")
        os.makedirs(self.af3_input_temp_base_dir, exist_ok=True)

        if not os.path.exists(self.executable_path):
            logger.error(f"AlphaFold3 executable not found at: {self.executable_path}. Please check config.yaml.")
            self.enabled = False
        elif not os.path.exists(self.db_dir) or not os.path.exists(self.model_dir):
            logger.error(
                f"AlphaFold3 database ({self.db_dir}) or model ({self.model_dir}) directories not found. AF3 disabled.")
            self.enabled = False
        else:
            self.enabled = self.config.enabled
            logger.info(f"AlphaFold3 Predictor initialized. Enabled: {self.enabled}")

    def _prepare_af3_input_json_batch(self, binders_data_list, batch_run_identifier):
        """
        Prepares a directory containing multiple input JSON files for a batch AF3 run.

        Args:
            binders_data_list (list): List of dictionaries, each describing a binder candidate.
            batch_run_identifier (str): Unique identifier for this entire batch AF3 run.

        Returns:
            str: Path to the created input directory containing all JSONs.
            None: If any critical input file (like .a3m) is missing for a target.
        """
        logger.debug(f"Preparing AF3 input JSONs for batch run: {batch_run_identifier}")

        # Create a unique input directory for this entire batch AF3 run
        current_batch_input_dir = os.path.join(self.af3_input_temp_base_dir, batch_run_identifier)
        os.makedirs(current_batch_input_dir, exist_ok=True)

        for i, binder_entry in enumerate(binders_data_list):
            target_id = binder_entry['target_id']
            target_sequence = binder_entry['target_sequence']
            binder_sequence = binder_entry['binder_sequence']
            binder_unique_id = binder_entry['id']

            if not os.path.exists(self.target_msa_dir):
                logger.error(
                    f"Target MSA (.a3m) file not found for target {target_id} at: {self.target_msa_dir}. Skipping this binder.")
                # We can't proceed with AF3 for this specific binder, but the batch might continue
                # The orchestrator will need to check for "failed_input_prep" status later.
                binder_entry.update({
                    "af3_status": "failed_input_prep",
                    "iptm": None,
                    "ptm": None,
                    'ranking_score': None,
                    'binder_plddt': None,
                    'target_plddt': None,
                    'pae': None,
                    'sc': None,
                    'surface_hydrophobicity': None,
                    'interface_delta_unsat_hbonds': None,
                    'complex_plddt': None,
                    "interface_hbonds": None,
                    "num_interface_resi": None,
                    "is_in_defined_binding_region": None,
                    "af3_pdb_path": None,
                    'converted_pdb_path': None,
                    "af3_output_dir": None,
                    "af3_error": f"Target MSA not found: {self.target_msa_dir}"
                })
                continue  # Skip to the next binder in the list

            # Each complex gets its own JSON file
            json_file_name = f"complex_{binder_unique_id}.json"  # Unique name for each JSON
            json_file_path = os.path.join(current_batch_input_dir, json_file_name)

            json_data = {
                "dialect": "alphafold3",
                "version": 3,
                "name": f"{binder_unique_id}",  # Name used by AF3 internally
                "modelSeeds": [3],  # Can be configured via config if needed
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": target_sequence,
                            "modifications": [],
                            "unpairedMsaPath": self.target_msa_dir,
                            "pairedMsa": "",
                            "templates": [
                                {
                                    "mmcifPath": self.target_cif_path,
                                    "queryIndices": list(range(len(target_sequence))),
                                    "templateIndices": list(range(len(target_sequence)))
                                }
                            ]
                        }
                    },
                    {
                        "protein": {
                            "id": "B",
                            "sequence": binder_sequence,
                            "modifications": [],
                            "unpairedMsa": f">query\n{binder_sequence}\n",
                            "pairedMsa": "",
                            "templates": []
                        }
                    }
                ]
            }

            try:
                with open(json_file_path, "w") as f:
                    json.dump(json_data, f, indent=2)
                logger.debug(f"Prepared AF3 input JSON for {binder_unique_id} at: {json_file_path}")
            except Exception as e:
                logger.error(f"Failed to write AF3 input JSON for {binder_unique_id}: {e}")
                # Mark this specific binder as failed for AF3 input prep
                binder_entry.update({
                    "af3_status": "failed_input_prep",
                    "iptm": None,
                    "ptm": None,
                    'ranking_score': None,
                    'binder_plddt': None,
                    'target_plddt': None,
                    'pae': None,
                    'sc': None,
                    'surface_hydrophobicity': None,
                    'interface_delta_unsat_hbonds': None,
                    'complex_plddt': None,
                    "interface_hbonds": None,
                    "num_interface_resi": None,
                    "is_in_defined_binding_region": None,
                    "af3_pdb_path": None,
                    'converted_pdb_path': None,
                    "af3_output_dir": None,
                    "af3_error": f"Failed to write JSON: {e}"
                })
                continue

        # Return the directory path where all the JSONs are stored
        return current_batch_input_dir

    def _parse_af3_output_for_complex(self, complex_output_dir):
        """
        Parses AF3 output for a single complex from its dedicated output directory.
        """
        iptm_score = None
        ptm_score = None
        binder_plddt = None
        target_plddt = None
        complex_plddt = None
        ranking_score = None
        pae_score = None
        pdb_path = None
        converted_pdb_path = None
        is_in_defined_binding_region = True
        num_interface_resi, sc, surface_hydrophobicity, interface_delta_unsat_hbonds, interface_hbonds = None, None, None, None, None
        prediction_subdir = Path(complex_output_dir)

        json_files = list(prediction_subdir.glob("*_confidences.json"))
        metrics_file, plddt_file = sorted(json_files, key=lambda s: '_summary' not in str(s))
        pdb_files = list(prediction_subdir.glob("*model.cif"))

        if json_files:
            try:
                with open(metrics_file, 'r') as f, open(plddt_file, 'r') as l:
                    metrics = json.load(f)
                    plddt_file = json.load(l)
                iptm_score = metrics.get('iptm')
                ptm_score = metrics.get('ptm')
                ranking_score = metrics.get('ranking_score')
                chain_pair_pae_min = metrics.get('chain_pair_pae_min')
                pae_score = max(max(row) for row in chain_pair_pae_min)
                chain_A_length = plddt_file.get('atom_chain_ids').count('A')
                chain_A_plddt = plddt_file.get('atom_plddts')[0:chain_A_length]
                chain_B_plddt = plddt_file.get('atom_plddts')[chain_A_length:len(plddt_file.get('atom_chain_ids'))]

                target_plddt = sum(chain_A_plddt) / len(chain_A_plddt)
                binder_plddt = sum(chain_B_plddt) / len(chain_B_plddt)
                complex_plddt = sum(plddt_file.get('atom_plddts')) / len(plddt_file.get('atom_plddts'))
            except Exception as e:
                logger.warning(f"Failed to parse metrics from {metrics_file}: {e}")

        if pdb_files:
            pdb_path = str(pdb_files[0])
            converted_pdb_path = pdb_path.replace('.cif', '.pdb')
            pymol_cmd.load(pdb_path)
            object_name = pymol_cmd.get_object_list('(all)')[0]
            pymol_cmd.extract("cleaned_object", object_name)
            pymol_cmd.save(converted_pdb_path, "cleaned_object")
            pymol_cmd.delete("all")  # Clean up
            sc, surface_hydrophobicity, interface_delta_unsat_hbonds, interface_hbonds = self._pyrosetta(
                converted_pdb_path)
            num_interface_resi = interface_resi_num(converted_pdb_path,
                                                    distance_threshold=self.global_config.filters.complex_filters.distance_for_counting_interface, )
            if self.global_config.models.seq2seq.define_binding.enabled:
                binding_region_str = self.global_config.models.seq2seq.define_binding.binding_region
                contact_resn_target_chain = interface_resi_num(converted_pdb_path,'A','B')
                is_in_defined_binding_region = validate_binder_contacts(contact_resn_target_chain,binding_region_str)

        return {
            "af3_status": "success",
            "iptm": iptm_score,
            "ptm": ptm_score,
            'binder_plddt': round(binder_plddt / 100, 3),
            'target_plddt': round(target_plddt / 100, 3),
            'complex_plddt': round(complex_plddt / 100, 3),
            'pae': pae_score,
            'shape_complementarity': sc,
            'surface_hydrophobicity': surface_hydrophobicity,
            'interface_delta_unsat_hbonds': interface_delta_unsat_hbonds,
            'interface_hbonds': interface_hbonds,
            'num_interface_resi': len(num_interface_resi),
            'is_in_defined_binding_region': is_in_defined_binding_region,
            'ranking_score': ranking_score,
            "af3_pdb_path": pdb_path,
            'converted_pdb_path': converted_pdb_path,
            "af3_output_dir": str(complex_output_dir),  # Path to this specific complex's output folder
            "af3_error": None
        }

    def _pyrosetta(self, pdb):
        init(extra_options='-holes:dalphaball src/models/DAlphaBall.gcc')
        pose = pose_from_pdb(pdb)

        iam = InterfaceAnalyzerMover(1, True)
        iam.set_compute_interface_sc(True)

        iam.apply(pose)

        # Extract Shape Complementarity
        iface_data = iam.get_all_data()
        sc = iface_data.sc_value
        interface_hbonds = iface_data.interface_hbonds  # number of interface H-bonds

        # calculate surface hydrophobicity
        binder_pose = {pose.pdb_info().chain(pose.conformation().chain_begin(i)): p for i, p in
                       zip(range(1, pose.num_chains() + 1), pose.split_by_chain())}["B"]

        layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
        layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
        surface_res = layer_sel.apply(binder_pose)

        exp_apol_count = 0
        total_count = 0

        # count apolar and aromatic residues at the surface
        for i in range(1, len(surface_res) + 1):
            if surface_res[i] == True:
                res = binder_pose.residue(i)

                # count apolar and aromatic residues as hydrophobic
                if res.is_apolar() == True or res.name() == 'PHE' or res.name() == 'TRP' or res.name() == 'TYR':
                    exp_apol_count += 1
                total_count += 1

        surface_hydrophobicity = exp_apol_count / total_count

        buns_filter = XmlObjects.static_get_filter(
            '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />')
        interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

        return sc, surface_hydrophobicity, interface_delta_unsat_hbonds, interface_hbonds

    def predict_complexes(self, binders_data_list, iteration_num):
        """
        Runs AlphaFold3 to predict complex structures for a batch of binders.

        Args:
            binders_data_list (list): A list of dictionaries, each representing
                                      a generated binder and its associated target info.
            iteration_num (int): The current active learning iteration number.

        Returns:
            list: The input list of dictionaries, with each binder's entry updated
                  with AF3 prediction results and status.
        """
        if not self.enabled:
            logger.debug("AlphaFold3 is disabled. Skipping all predictions.")
            for binder_entry in binders_data_list:
                binder_entry.update({
                    "af3_status": "skipped",
                    "iptm": None,
                    "ptm": None,
                    'ranking_score': None,
                    'binder_plddt': None,
                    'target_plddt': None,
                    'pae': None,
                    'sc': None,
                    'surface_hydrophobicity': None,
                    'interface_delta_unsat_hbonds': None,
                    'complex_plddt': None,
                    "interface_hbonds": None,
                    "num_interface_resi": None,
                    "is_in_defined_binding_region": None,
                    "af3_pdb_path": None,
                    'converted_pdb_path': None,
                    "af3_output_dir": None,
                    "af3_error": "AF3 disabled"
                })
            return binders_data_list

        if not binders_data_list:
            logger.warning("No binders provided for AF3 prediction.")
            return []

        # --- Generate a unique identifier for this entire batch AF3 run ---
        batch_run_identifier = f"iteration_{iteration_num}_batch_{int(time.time())}"

        # --- 1. Prepare ALL AlphaFold3 Input JSONs for the batch ---
        af3_input_json_dir = self._prepare_af3_input_json_batch(
            binders_data_list, batch_run_identifier
        )
        if af3_input_json_dir is None:  # This would happen if _prepare_af3_input_json_batch decided to return None (e.g., critical error)
            logger.error("Failed to prepare AF3 batch input JSONs. No AF3 predictions will be run.")
            # Ensure all binders are marked as failed if input prep failed for the whole batch
            for binder_entry in binders_data_list:
                if binder_entry.get('af3_status') != "failed_input_prep":  # Don't overwrite existing failures
                    binder_entry.update(
                        {"af3_status": "failed_batch_input_prep",
                         "iptm": None,
                         "ptm": None,
                         'ranking_score': None,
                         "af3_pdb_path": None,
                         'converted_pdb_path': None,
                         'binder_plddt': None,
                         'target_plddt': None,
                         'pae': None,
                         'sc': None,
                         'surface_hydrophobicity': None,
                         'interface_delta_unsat_hbonds': None,
                         'complex_plddt': None,
                         "interface_hbonds": None,
                         "num_interface_resi": None,
                         "is_in_defined_binding_region": None,
                         "af3_output_dir": None,
                         "af3_error": "Failed batch input JSON prep"})
            return binders_data_list

        # --- 2. Define the SINGLE AlphaFold3 Output Directory for this batch ---
        # AF3 will create subdirectories for *each* complex inside this
        af3_batch_output_root_dir = os.path.join(self.project_af3_output_base_dir, f"output_{batch_run_identifier}")
        os.makedirs(af3_batch_output_root_dir, exist_ok=True)

        # --- 3. Construct and Execute the SINGLE AlphaFold3 command for the batch ---
        cmd = [
            "taskset", "-c", f"{self.cpu}",
            "env", f"CUDA_VISIBLE_DEVICES={self.gpu}",
            "mamba", "run", "-n", self.conda_env_name,
            "python",
            self.executable_path,
            "--db_dir", self.db_dir,
            "--num_diffusion_samples", str(self.config.num_diffusion_samples),
            "--force_output_dir", str(self.config.force_output_dir),
            "--input_dir", af3_input_json_dir,  # Point to the directory containing all JSONs
            "--jackhmmer_n_cpu", str(self.config.jackhmmer_n_cpu),
            "--max_template_date", str(self.config.max_template_date),
            "--model_dir", self.model_dir,
            "--nhmmer_n_cpu", str(self.config.nhmmer_n_cpu),
            "--output_dir", af3_batch_output_root_dir,  # AF3's root output directory for this batch
        ]

        logger.info(f"Executing AlphaFold3 for batch '{batch_run_identifier}'. Command: {' '.join(cmd)}")

        try:
            af3_run_start_time = time.time()
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            af3_run_end_time = time.time()

            logger.debug(f"AF3 batch stdout:\n{process.stdout}")
            logger.debug(f"AF3 batch stderr:\n{process.stderr}")
            logger.info(
                f"AlphaFold3 finished batch prediction in {round(af3_run_end_time - af3_run_start_time, 2)} seconds.")

            # --- 4. Parse AlphaFold3 Output for EACH Complex ---
            # AF3 creates a subdirectory for each complex within af3_batch_output_root_dir
            # The name of this subdirectory typically matches the 'name' field in the JSON file
            # or the name of the JSON file itself.

            updated_binders_data = []

            # Iterate through the original binders_data_list to update them
            # We need to map the output subdirectories back to the correct binder_entry

            # A common way AF3 names output subdirs is after the input JSON filename (without .json)
            # Or after the 'name' field within the JSON. Let's assume the latter for now.
            # So, for complex_UNIQUE_ID.json, AF3 creates a folder named complex_UNIQUE_ID/

            logger.info("Parsing AlphaFold3 outputs for individual complexes...")
            for binder_entry in tqdm(binders_data_list, desc="Parsing AF3 Outputs", unit="complex"):
                # If this binder was already marked as failed input prep, skip AF3 output parsing
                if binder_entry.get("af3_status") == "failed_input_prep":
                    updated_binders_data.append(binder_entry)
                    continue

                # The output directory for this specific complex within the batch output root
                complex_output_dir_name = binder_entry['id'].lower()
                complex_output_dir_path = Path(
                    af3_batch_output_root_dir) / complex_output_dir_name  # Adjusted based on json_file_name

                if complex_output_dir_path.is_dir():
                    single_complex_results = self._parse_af3_output_for_complex(complex_output_dir_path)
                    binder_entry.update(single_complex_results)
                    binder_entry["af3_prediction_time_s"] = round(
                        (af3_run_end_time - af3_run_start_time) / len(binders_data_list), 2)  # Avg time
                else:
                    logger.warning(
                        f"Output directory not found for complex {complex_output_dir_name} at {complex_output_dir_path}. AF3 prediction might have failed for it.")
                    binder_entry.update({
                        "af3_status": "failed_output_parse",
                        "iptm": None,
                        "ptm": None,
                        'ranking_score': None,
                        'binder_plddt': None,
                        'target_plddt': None,
                        'pae': None,
                        'sc': None,
                        'surface_hydrophobicity': None,
                        'interface_delta_unsat_hbonds': None,
                        'complex_plddt': None,
                        "interface_hbonds": None,
                        "num_interface_resi": None,
                        "is_in_defined_binding_region": None,
                        "af3_pdb_path": None,
                        'converted_pdb_path': None,
                        "af3_output_dir": str(complex_output_dir_path),
                        "af3_error": "Output directory not found after batch run."
                    })
                updated_binders_data.append(binder_entry)

            return updated_binders_data

        except subprocess.CalledProcessError as e:
            logger.error(
                f"AlphaFold3 batch command failed (Exit Code: {e.returncode}). Stdout: {e.stdout}. Stderr: {e.stderr}")
            # Mark all binders in the current batch as failed if the main AF3 process itself failed
            for binder_entry in binders_data_list:
                binder_entry.update({
                    "af3_status": "failed_batch_subprocess",
                    "iptm": None,
                    "ptm": None,
                    'ranking_score': None,
                    'binder_plddt': None,
                    'target_plddt': None,
                    'pae': None,
                    'sc': None,
                    'surface_hydrophobicity': None,
                    'interface_delta_unsat_hbonds': None,
                    'complex_plddt': None,
                    "interface_hbonds": None,
                    "num_interface_resi": None,
                    "is_in_defined_binding_region": None,
                    "af3_pdb_path": None,
                    'converted_pdb_path': None,
                    "af3_output_dir": af3_batch_output_root_dir,
                    "af3_error": f"AF3 batch subprocess error: {e.returncode}"
                })
            return binders_data_list
        except FileNotFoundError:
            logger.error(f"AlphaFold3 executable '{self.executable_path}' not found or not in PATH.")
            for binder_entry in binders_data_list:
                binder_entry.update({
                    "af3_status": "failed_executable",
                    "iptm": None,
                    "ptm": None,
                    'ranking_score': None,
                    'binder_plddt': None,
                    'target_plddt': None,
                    'pae': None,
                    'sc': None,
                    'surface_hydrophobicity': None,
                    'interface_delta_unsat_hbonds': None,
                    'complex_plddt': None,
                    "interface_hbonds": None,
                    "num_interface_resi": None,
                    "is_in_defined_binding_region": None,
                    "af3_pdb_path": None,
                    'converted_pdb_path': None,
                    "af3_output_dir": af3_batch_output_root_dir,
                    "af3_error": "AF3 executable not found"
                })
            return binders_data_list
        except Exception as e:
            logger.error(f"An unexpected error occurred during AF3 batch prediction: {e}")
            for binder_entry in binders_data_list:
                binder_entry.update({
                    "af3_status": "failed_general",
                    "iptm": None,
                    "ptm": None,
                    'ranking_score': None,
                    'binder_plddt': None,
                    'target_plddt': None,
                    'pae': None,
                    'sc': None,
                    'surface_hydrophobicity': None,
                    'interface_delta_unsat_hbonds': None,
                    'complex_plddt': None,
                    "interface_hbonds": None,
                    "num_interface_resi": None,
                    "is_in_defined_binding_region": None,
                    "af3_pdb_path": None,
                    'converted_pdb_path': None,
                    "af3_output_dir": af3_batch_output_root_dir,
                    "af3_error": f"Unexpected error: {e}"
                })
            return binders_data_list
