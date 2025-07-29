# src/main.py
import logging
import os
import torch
import argparse
from omegaconf import OmegaConf # Import OmegaConf for saving
from src.utils.helpers import load_config,setup_logging
from src.data_access.manager import DataManager
from src.models.generator import BinderGenerator
from src.core.loop import ActiveLearningOrchestrator

logger = logging.getLogger(__name__)


def run_bindsculpt(resume_from_iter: int = 0):  # <--- NEW: Add argument with default
    """
    Main entry point function for the BindSculpt application.
    Initializes configuration and starts the active learning orchestration.
    """
    logger.info("BindSculpt application started.")
    if resume_from_iter > 0:
        logger.info(f"--- RESUMING RUN FROM ITERATION {resume_from_iter} ---")
    try:
        config = load_config()

        setup_logging(
            log_level=config.log_level,
            log_dir=config.paths.log_dir  # Use the resolved path from the config
        )

        if config is None:
            logger.critical("Failed to load configuration. Exiting BindSculpt.")
            return

        logger.info("Configuration loaded successfully.")

        # Ensure the output directory exists before saving
        os.makedirs(config.paths.run_output_dir, exist_ok=True)
        config_save_path = os.path.join(config.paths.run_output_dir, 'run_config.yaml')
        try:
            OmegaConf.save(config, config_save_path)
            logger.info(f"Saved final run configuration to {config_save_path}")
        except Exception as e:
            logger.error(f"Failed to save run configuration: {e}")

        orchestrator = ActiveLearningOrchestrator(config)

        # --- NEW: Pass the resume iteration to the main loop ---
        orchestrator.run_full_active_learning_loop(start_iteration=resume_from_iter)

        logger.info("BindSculpt application finished.")

    except Exception as e:
        logger.critical(f"An unhandled critical error occurred during BindSculpt execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_bindsculpt(resume_from_iter=0)

    ## if use run_bindsculpt alone:
    # parser = argparse.ArgumentParser(description="Run the BindSculpt Active Learning pipeline.")
    # parser.add_argument(
    #     '--resume_from_iter',
    #     type=int,
    #     default=0,
    #     required=True,
    #     help='The iteration number to resume from. Default is 0 (start from scratch with Iteration 0 generation).'
    # )
    # args = parser.parse_args()

    # Call the main function with the parsed argument
    # run_bindsculpt(resume_from_iter=args.resume_from_iter)