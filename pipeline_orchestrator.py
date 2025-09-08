#!/usr/bin/env python3
"""
Distributed Pipeline Orchestrator

This script orchestrates the complete distributed pipeline:
1. Training (optional)
2. Sampling (optional)
3. Evaluation (optional)

It handles:
- Configuration management
- Inter-stage data passing
- Distributed execution
- Error handling and recovery
- Progress tracking
"""

import os
import sys
import json
import time
import glob
import shutil
import signal
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_config import (
    PipelineConfig, DEFAULT_CONFIG, CONFIGS,
    TrainingConfig, SamplingConfig, EvaluationConfig
)
from config_loader import ConfigLoader, ConfigurationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_af2_mode_enabled_static(additional_args: List[str]) -> bool:
    """
    Static version of AF2 mode detection that doesn't require a class instance.
    This prevents NameError issues when called from main() function.
    
    Args:
        additional_args: List of additional training arguments
        
    Returns:
        bool: True if AF2 mode is enabled, False otherwise
    """
    if not isinstance(additional_args, list):
        raise TypeError(f"additional_args must be a list, got {type(additional_args)}")
    
    additional_args_str = ' '.join(additional_args)
    
    # Check for explicit --use_af2 flag (backward compatibility)
    if '--use_af2' in additional_args_str:
        return True
        
    # Check for ratio_af2_pdb != 0 (new hybrid mode detection)
    for i, arg in enumerate(additional_args):
        if arg == '--ratio_af2_pdb' and i + 1 < len(additional_args):
            try:
                ratio_value = int(additional_args[i + 1])
                if ratio_value != 0:
                    return True
            except (ValueError, IndexError):
                pass  # Invalid ratio value, continue checking
                
    return False

class PipelineOrchestrator:
    """Main orchestrator class for the distributed pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.start_time = time.time()
        self.stage_results = {}
        self.current_stage = None
        self.processes = []
        
        # Set up output directory
        self.setup_output_directory()
        
        # Set up logging
        self.setup_logging()
        
        # Register signal handlers for cleanup
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_output_directory(self):
        """Set up the main output directory."""
        # Only create directory if we have write permissions
        try:
            if not os.path.exists(self.config.base_output_dir):
                os.makedirs(self.config.base_output_dir, exist_ok=True)
        except PermissionError:
            logger.warning(f"Cannot create output directory {self.config.base_output_dir} - assuming it will be created by AMLT")
        
        # Resolve stage-specific output directories relative to base output directory
        self.resolve_stage_output_directories()
        
        # Try to save configuration, but don't fail if we can't write
        try:
            config_path = os.path.join(self.config.base_output_dir, self.config.pipeline_config_save_filename)
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
        except (PermissionError, FileNotFoundError):
            logger.warning(f"Cannot write configuration file to {self.config.base_output_dir} - continuing without saving config")
        
        logger.info(f"Pipeline output directory: {self.config.base_output_dir}")
    
    def resolve_stage_output_directories(self):
        """Resolve stage output directories relative to base output directory."""
        # Debug logging
        logger.info(f"Resolving stage directories. base_output_dir: {self.config.base_output_dir}")
        logger.info(f"Training output_dir before resolution: {self.config.training.output_dir}")
        
        # Only resolve if the paths are relative
        if not os.path.isabs(self.config.training.output_dir):
            self.config.training.output_dir = os.path.join(
                self.config.base_output_dir, self.config.training.output_dir
            )
            logger.info(f"Training output_dir after resolution: {self.config.training.output_dir}")
        
        if not os.path.isabs(self.config.sampling.output_dir):
            self.config.sampling.output_dir = os.path.join(
                self.config.base_output_dir, self.config.sampling.output_dir
            )
        
        if not os.path.isabs(self.config.evaluation.output_dir):
            self.config.evaluation.output_dir = os.path.join(
                self.config.base_output_dir, self.config.evaluation.output_dir
            )
    
    def setup_logging(self):
        """Set up pipeline-specific logging."""
        log_file = os.path.join(self.config.base_output_dir, "pipeline.log")
        
        # Try to create file handler, but don't fail if we can't write
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
            
            logger.info(f"Pipeline logging to: {log_file}")
        except (PermissionError, FileNotFoundError):
            logger.warning(f"Cannot write log file to {log_file} - logging only to console")
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals for cleanup."""
        logger.warning(f"Received signal {signum}, cleaning up...")
        self.cleanup()
        sys.exit(1)
    
    def cleanup(self):
        """Clean up resources and processes."""
        logger.info("Cleaning up pipeline resources...")
        
        # Terminate any running processes
        for proc in self.processes:
            if proc.poll() is None:  # Process is still running
                logger.info(f"Terminating process {proc.pid}")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing process {proc.pid}")
                    proc.kill()
    
    def copy_key_results_to_output_dir(self, stage_name: str, stage_output_dir: str):
        """
        Copy key result files from stage output directory to main output directory.
        This ensures important results are available in the AMLT output location.
        
        Args:
            stage_name: Name of the stage (training, sampling, evaluation)
            stage_output_dir: The stage-specific output directory (/tmp/sampling, etc.)
        """
        if not os.path.exists(stage_output_dir):
            logger.warning(f"Stage output directory {stage_output_dir} does not exist, skipping copy")
            return
        
        # Define key files to copy for each stage
        key_files = {
            "sampling": [
                "*_sequences.csv",  # Main output CSV with sequences
                "*_metadata.txt",   # Sampling metadata
                "*_probabilities.npz"  # Probabilities (optional, but small enough to include)
            ],
            "evaluation": [
                "evaluation_summary.txt", 
                "prediction_log.txt",
                "structure_comparison_results.csv"
            ],
            "training": [
                "*.log",  # Training logs
                "best_model.pth",  # Best model checkpoint (if exists)
                "final_model.pth"  # Final model checkpoint (if exists),
                "*best.pt"  # Final model checkpoint (if exists)
            ]
        }
        
        files_to_copy = key_files.get(stage_name, [])
        if not files_to_copy:
            logger.debug(f"No key files defined for stage {stage_name}")
            return
        
        copied_files = []
        for file_pattern in files_to_copy:
            # Handle glob patterns
            if '*' in file_pattern:
                pattern_path = os.path.join(stage_output_dir, file_pattern)
                matching_files = glob.glob(pattern_path)
                
                for src_path in matching_files:
                    filename = os.path.basename(src_path)
                    try:
                        # Create destination subdirectory for organization
                        dest_dir = os.path.join(self.config.base_output_dir, f"{stage_name}_results")
                        os.makedirs(dest_dir, exist_ok=True)
                        
                        dest_path = os.path.join(dest_dir, filename)
                        shutil.copy2(src_path, dest_path)
                        copied_files.append(filename)
                        logger.info(f"Copied {filename} to {dest_dir}")
                    except (PermissionError, OSError) as e:
                        logger.warning(f"Failed to copy {filename}: {e}")
            else:
                # Handle exact filename
                src_path = os.path.join(stage_output_dir, file_pattern)
                if os.path.exists(src_path):
                    try:
                        # Create destination subdirectory for organization
                        dest_dir = os.path.join(self.config.base_output_dir, f"{stage_name}_results")
                        os.makedirs(dest_dir, exist_ok=True)
                        
                        dest_path = os.path.join(dest_dir, file_pattern)
                        shutil.copy2(src_path, dest_path)
                        copied_files.append(file_pattern)
                        logger.info(f"Copied {file_pattern} to {dest_dir}")
                    except (PermissionError, OSError) as e:
                        logger.warning(f"Failed to copy {file_pattern}: {e}")
                else:
                    logger.debug(f"Key file {file_pattern} not found in {stage_output_dir}")
        
        if copied_files:
            logger.info(f"Copied {len(copied_files)} key result files from {stage_name} to main output directory")
        else:
            logger.warning(f"No key result files found to copy from {stage_name}")
    
    def run_command(self, command: List[str], stage_name: str, 
                   capture_output: bool = False, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        """
        Run a command with proper logging and error handling.
        
        Args:
            command: Command to run as list of strings
            stage_name: Name of the current stage for logging
            capture_output: Whether to capture stdout/stderr
            timeout: Command timeout in seconds
            
        Returns:
            CompletedProcess result
        """
        logger.info(f"[{stage_name}] Running command: {' '.join(command)}")
        
        # Create log files for this stage if we have write permissions
        stage_log_dir = os.path.join(self.config.base_output_dir, "logs", stage_name)
        try:
            os.makedirs(stage_log_dir, exist_ok=True)
            stdout_file = os.path.join(stage_log_dir, "stdout.log")
            stderr_file = os.path.join(stage_log_dir, "stderr.log")
            write_logs = True
        except PermissionError:
            logger.warning(f"Cannot create log directory {stage_log_dir} - logs will not be saved")
            write_logs = False
        
        try:
            if capture_output:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
            else:
                if write_logs:
                    with open(stdout_file, 'w') as stdout_f, open(stderr_file, 'w') as stderr_f:
                        result = subprocess.run(
                            command,
                            stdout=stdout_f,
                            stderr=stderr_f,
                            text=True,
                            timeout=timeout,
                            cwd=os.path.dirname(os.path.abspath(__file__))
                        )
                else:
                    # Run without capturing logs to files
                    result = subprocess.run(
                        command,
                        text=True,
                        timeout=timeout,
                        cwd=os.path.dirname(os.path.abspath(__file__))
                    )
            
            if result.returncode == 0:
                logger.info(f"[{stage_name}] Command completed successfully")
            else:
                logger.error(f"[{stage_name}] Command failed with return code {result.returncode}")
                
                # Log error details
                if capture_output and result.stderr:
                    logger.error(f"[{stage_name}] Error output: {result.stderr}")
                elif os.path.exists(stderr_file):
                    with open(stderr_file, 'r') as f:
                        error_content = f.read()
                        if error_content:
                            logger.error(f"[{stage_name}] Error output: {error_content}")
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.error(f"[{stage_name}] Command timed out after {timeout} seconds")
            raise
        except Exception as e:
            logger.error(f"[{stage_name}] Command execution failed: {e}")
            raise
    
    def find_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """
        Find the latest checkpoint file in a directory.
        Also attempts to copy checkpoints to local tmp for sampling if needed.
        """
        logger.info(f"[DEBUG] Looking for checkpoints in: {checkpoint_dir}")
        print(f"[DEBUG] Looking for checkpoints in: {checkpoint_dir}", flush=True)
        
        if not os.path.exists(checkpoint_dir):
            logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
            print(f"[DEBUG] Directory {checkpoint_dir} does not exist", flush=True)
            return None
        
        # List all files in the directory for debugging
        all_files = os.listdir(checkpoint_dir)
        logger.info(f"[DEBUG] Files in {checkpoint_dir}: {all_files}")
        print(f"[DEBUG] Files in {checkpoint_dir}: {all_files}", flush=True)
        
        checkpoint_files = []
        for file in all_files:
            if file.endswith(('.pt', '.pth', '.ckpt')):
                file_path = os.path.join(checkpoint_dir, file)
                checkpoint_files.append((file_path, os.path.getmtime(file_path)))
                logger.info(f"[DEBUG] Found checkpoint: {file}")
                print(f"[DEBUG] Found checkpoint: {file}", flush=True)
        
        if not checkpoint_files:
            logger.warning(f"No checkpoint files found in: {checkpoint_dir}")
            print(f"[DEBUG] No .pt/.pth/.ckpt files found in {checkpoint_dir}", flush=True)
            return None
        
        # Return the most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: x[1])[0]
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        print(f"[DEBUG] Latest checkpoint: {latest_checkpoint}", flush=True)
        
        # Copy checkpoint to tmp for sampling accessibility
        tmp_path = self._ensure_checkpoint_in_tmp(latest_checkpoint)
        if tmp_path:
            print(f"[DEBUG] Successfully copied checkpoint to tmp: {tmp_path}", flush=True)
        
        return latest_checkpoint
    
    def _ensure_checkpoint_in_tmp(self, checkpoint_path: str) -> Optional[str]:
        """
        Copy checkpoint to ../tmp directory for sampling accessibility.
        Returns the tmp path if successful, None otherwise.
        """
        try:
            # Create tmp directory structure
            tmp_base = os.path.abspath(self.config.tmp_checkpoint_dir)
            os.makedirs(tmp_base, exist_ok=True)
            
            checkpoint_filename = os.path.basename(checkpoint_path)
            tmp_checkpoint_path = os.path.join(tmp_base, checkpoint_filename)
            
            # Copy checkpoint to tmp
            import shutil
            shutil.copy2(checkpoint_path, tmp_checkpoint_path)
            logger.info(f"Copied checkpoint to tmp: {tmp_checkpoint_path}")
            
            return tmp_checkpoint_path
        except Exception as e:
            logger.warning(f"Failed to copy checkpoint to tmp: {e}")
            return None
    
    def _get_process_seed(self) -> int:
        """
        Generate a unique seed for this orchestrator run to ensure distributed processes
        don't sample the same data.
        
        Returns:
            int: Unique seed based on timestamp and process ID
        """
        import time
        # Create a unique seed based on current time and process ID
        # This ensures that even if multiple orchestrator instances start simultaneously,
        # they will have different seeds
        base_seed = int(time.time() * 1000) % (2**31 - 1)  # Use milliseconds, keep within int32 range
        process_seed = (base_seed + os.getpid()) % (2**31 - 1)  # Add process ID for uniqueness
        
        logger.info(f"Generated process seed: {process_seed} (base: {base_seed}, pid: {os.getpid()})")
        return process_seed
    
    
    def is_af2_mode_enabled(self, additional_args: List[str]) -> bool:
        """
        Check if AF2 mode is enabled based on either --use_af2 flag or ratio_af2_pdb != 0.
        
        Args:
            additional_args: List of additional training arguments
            
        Returns:
            bool: True if AF2 mode is enabled, False otherwise
        """
        additional_args_str = ' '.join(additional_args)
        
        # Check for explicit --use_af2 flag (backward compatibility)
        if '--use_af2' in additional_args_str:
            return True
            
        # Check for ratio_af2_pdb != 0 (new hybrid mode detection)
        for i, arg in enumerate(additional_args):
            if arg == '--ratio_af2_pdb' and i + 1 < len(additional_args):
                try:
                    ratio_value = int(additional_args[i + 1])
                    if ratio_value != 0:
                        return True
                except (ValueError, IndexError):
                    pass  # Invalid ratio value, continue checking
                    
        return False
    
    def validate_af2_configuration(self, config: TrainingConfig) -> bool:
        """
        Validate AF2 configuration if AF2 mode is enabled.
        
        Args:
            config: Training configuration to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check if AF2 mode is enabled
        if not self.is_af2_mode_enabled(config.additional_args):
            return True  # Not using AF2, no validation needed
        
        logger.info("AF2 mode detected, validating AF2 configuration...")
        
        # Check for AF2 data source: chunk directory
        af2_chunk_dir = None
        args = config.additional_args
        
        for i, arg in enumerate(args):
            if arg == '--af2_chunk_dir' and i + 1 < len(args):
                af2_chunk_dir = args[i + 1]
                break
        
        # AF2 mode requires chunk directory
        if not af2_chunk_dir:
            logger.warning("AF2 mode enabled but no --af2_chunk_dir specified")
            return False
        
        # For remote data paths, skip directory existence check as they may be AMLT variables
        # that aren't resolvable at validation time
        if af2_chunk_dir.startswith('$$') or af2_chunk_dir.startswith('/mnt/'):
            logger.info(f"AF2 chunk directory configured (remote path): {af2_chunk_dir}")
        else:
            # Validate local chunk directory exists
            if not os.path.exists(af2_chunk_dir):
                logger.error(f"AF2 chunk directory does not exist: {af2_chunk_dir}")
                return False
            logger.info(f"AF2 chunk directory validated (local path): {af2_chunk_dir}")
        
        return True
    
    def find_latest_output_file(self, output_dir: str, pattern: str) -> Optional[str]:
        """Find the latest output file matching a pattern."""
        if not os.path.exists(output_dir):
            return None
        
        matching_files = []
        for file in Path(output_dir).glob(pattern):
            matching_files.append((str(file), file.stat().st_mtime))
        
        if not matching_files:
            return None
        
        # Return the most recent file
        latest_file = max(matching_files, key=lambda x: x[1])[0]
        logger.info(f"Found latest output file: {latest_file}")
        return latest_file
    
    def run_training_stage(self) -> bool:
        """Run the training stage."""
        if not self.config.training.enabled:
            logger.info("Training stage disabled, skipping...")
            return True
        
        self.current_stage = "training"
        logger.info("=" * 60)
        logger.info("STARTING TRAINING STAGE")
        logger.info("=" * 60)
        
        config = self.config.training
        
        # Validate AF2 configuration if AF2 mode is enabled
        if not self.validate_af2_configuration(config):
            logger.error("AF2 configuration validation failed")
            return False
        
        # Prepare training command
        if config.distributed.enabled:
            command = [
                "torchrun",
                f"--nproc_per_node={config.distributed.num_gpus}",
                f"--master_port={config.distributed.master_port}",
                "training/train.py"
            ]
        else:
            command = ["python", "training/train.py"]
        
        # Filter out arguments that train.py doesn't support
        # Note: train.py now supports --verbose argument (as of recent updates)
        filtered_additional_args = []
        skip_next = False
        for i, arg in enumerate(config.additional_args):
            if skip_next:
                skip_next = False
                continue
            # Note: --verbose is now supported by train.py, so we don't filter it out
            filtered_additional_args.append(arg)
        
        # Add additional arguments first (these can be overridden)
        command.extend(filtered_additional_args)
        
        # Add/override critical orchestrator-managed arguments
        # These are added last to ensure they take precedence
        critical_args = [
            "--batch", str(config.batch_size),
            "--epochs", str(config.epochs),
            "--lr", str(config.learning_rate),
            "--output_dir", self.config.base_output_dir,  # Pass base output dir for model copying
        ]
        
        # Add config file only if specified
        if hasattr(config, 'config_path') and config.config_path:
            critical_args.extend(["--config_file", config.config_path])
        
        # Add gradient clipping parameter if specified
        if hasattr(config, 'grad_clip'):
            critical_args.extend(["--grad_clip", str(config.grad_clip)])
        
        # Add dual checkpoint saving to configurable tmp directory
        tmp_dir = os.path.abspath(config.checkpoint_copy_dir or self.config.tmp_checkpoint_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        critical_args.extend(["--checkpoint_copy_dir", tmp_dir])
        
        # Add time conditioning parameters if specified
        if hasattr(config, 'time_integration'):
            critical_args.extend(["--time_integration", config.time_integration])
        if hasattr(config, 'use_time_conditioning') and not config.use_time_conditioning:
            critical_args.append("--disable_time_conditioning")
        if hasattr(config, 'use_virtual_node') and config.use_virtual_node:
            critical_args.append("--use_virtual_node")
        
        # Debug logging for output directory
        logger.info(f"Training command will use output_dir: {self.config.base_output_dir}")
        logger.info(f"Training working directory: {config.output_dir}")
        
        # Add global flags if set
        if self.config.verbose:
            critical_args.append("--verbose")
        if getattr(self.config, 'no_source_indicator', False):
            critical_args.append("--no_source_indicator")
        if getattr(self.config, 'max_edge_dist', None) is not None:
            critical_args.extend(["--max_edge_dist", str(self.config.max_edge_dist)])
        
        command.extend(critical_args)
        
        # Add seed if explicitly set in config (override any additional_args seed)
        if config.seed is not None:
            command.extend(["--seed", str(config.seed)])
        
        # Add essential dataset parameters (use config instead of hardcoded)
        command.extend([
            "--split_json", config.dataset.split_json,
            "--map_pkl", config.dataset.map_pkl
        ])
        
        # Add distributed training flag if enabled (override if needed)
        if config.distributed.enabled:
            command.append("--distributed")
        
        # Add wandb configuration (override if needed)
        if config.wandb_enabled:
            command.extend([
                "--wandb_project", config.wandb_project,
                "--use_wandb"
            ])
        
        # Add resume checkpoint if specified
        if config.resume_from_checkpoint:
            command.extend(["--resume_from_checkpoint", config.resume_from_checkpoint])
        
        # Note: train.py doesn't have a --verbose argument, so we skip it
        # Verbose output is controlled by the logging level
        
        try:
            # Run training
            result = self.run_command(command, "training", timeout=config.distributed.timeout_minutes * 60)
            
            if result.returncode == 0:
                logger.info("Training stage completed successfully")
                
                # Find the latest checkpoint (check both primary location and tmp)
                latest_checkpoint = self.find_latest_checkpoint(config.checkpoint_dir)
                if not latest_checkpoint:
                    # Also check tmp directory as backup
                    tmp_dir = os.path.abspath("../tmp")
                    if os.path.exists(tmp_dir):
                        logger.info("No checkpoint in primary location, checking tmp directory...")
                        latest_checkpoint = self.find_latest_checkpoint(tmp_dir)
                
                if latest_checkpoint:
                    self.stage_results["training"] = {
                        "status": "success",
                        "checkpoint": latest_checkpoint,
                        "output_dir": config.output_dir
                    }
                else:
                    logger.warning("No checkpoint found after training")
                    self.stage_results["training"] = {
                        "status": "success",
                        "checkpoint": None,
                        "output_dir": config.output_dir
                    }
                
                # Copy key result files to main output directory
                self.copy_key_results_to_output_dir("training", config.output_dir)
                
                return True
            else:
                logger.error("Training stage failed")
                self.stage_results["training"] = {"status": "failed", "return_code": result.returncode}
                return False
                
        except Exception as e:
            logger.error(f"Training stage crashed: {e}")
            self.stage_results["training"] = {"status": "crashed", "error": str(e)}
            return False
    
    def run_sampling_stage(self) -> bool:
        """Run the sampling stage."""
        if not self.config.sampling.enabled:
            logger.info("Sampling stage disabled, skipping...")
            return True
        
        self.current_stage = "sampling"
        logger.info("=" * 60)
        logger.info("STARTING SAMPLING STAGE")
        logger.info("=" * 60)
        
        config = self.config.sampling
        
        # Determine model checkpoint to use
        model_checkpoint = config.model_checkpoint
        if model_checkpoint == "auto":
            if "training" in self.stage_results and self.stage_results["training"]["status"] == "success":
                model_checkpoint = self.stage_results["training"]["checkpoint"]
                if not model_checkpoint:
                    logger.error("Training completed but no checkpoint found")
                    return False
                
                # Prefer tmp checkpoint for sampling if available
                tmp_dir = os.path.abspath(self.config.tmp_checkpoint_dir)
                if os.path.exists(tmp_dir):
                    checkpoint_name = os.path.basename(model_checkpoint)
                    tmp_checkpoint = os.path.join(tmp_dir, checkpoint_name)
                    if os.path.exists(tmp_checkpoint):
                        logger.info(f"Using tmp checkpoint for sampling: {tmp_checkpoint}")
                        model_checkpoint = tmp_checkpoint
                    else:
                        logger.info(f"Tmp checkpoint not found, using primary: {model_checkpoint}")
                        
            else:
                logger.error("Auto model checkpoint requested but training did not complete successfully")
                return False
        
        if not model_checkpoint or not os.path.exists(model_checkpoint):
            logger.error(f"Model checkpoint not found: {model_checkpoint}")
            return False
        
        # Validate GPU configuration early (fail-fast)
        if config.distributed.enabled:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.error("CUDA not available but distributed GPU training requested")
                    return False
                gpu_count = torch.cuda.device_count()
                if gpu_count < config.distributed.num_gpus:
                    logger.error(f"Requested {config.distributed.num_gpus} GPUs but only {gpu_count} available")
                    return False
                logger.info(f"GPU validation passed: {gpu_count} GPUs available, using {config.distributed.num_gpus}")
            except ImportError:
                logger.error("PyTorch not available but distributed training requested")
                return False
        
        # Prepare sampling command
        if config.distributed.enabled:
            command = [
                "torchrun",
                f"--nproc_per_node={config.distributed.num_gpus}",
                f"--master_port={config.distributed.master_port}",
                "training/sample.py"
            ]
        else:
            command = ["python", "training/sample.py"]
        
        # Add essential orchestrator-managed arguments
        command.extend([
            "--model_path", model_checkpoint,
            "--split_json", config.dataset.split_json,
            "--map_pkl", config.dataset.map_pkl,
            "--output_dir", config.output_dir,
        ])
        
        # Automatically add structured config parameters as command line arguments
        # This eliminates the need to duplicate everything in additional_args
        # Support both the dataclass field names and the intuitive JSON field names
        max_structures_value = None
        if hasattr(config, 'max_structures') and config.max_structures is not None:
            max_structures_value = config.max_structures
        elif hasattr(config, 'num_samples') and config.num_samples is not None:
            max_structures_value = config.num_samples
            
        if max_structures_value is not None:
            command.extend(["--max_structures", str(max_structures_value)])
        
        if hasattr(config, 'flow_temp') and config.flow_temp is not None:
            command.extend(["--flow_temp", str(config.flow_temp)])
        elif hasattr(config, 'temperature') and config.temperature is not None:
            command.extend(["--flow_temp", str(config.temperature)])
            
        if hasattr(config, 'steps') and config.steps is not None:
            command.extend(["--steps", str(config.steps)])
            
        if hasattr(config, 'T') and config.T is not None:
            command.extend(["--T", str(config.T)])
            
        if hasattr(config, 't_min') and config.t_min is not None:
            command.extend(["--t_min", str(config.t_min)])
            
        if hasattr(config, 'split') and config.split is not None:
            command.extend(["--split", config.split])
            
        if hasattr(config, 'integration_method') and config.integration_method is not None:
            command.extend(["--integration_method", config.integration_method])
            
        if hasattr(config, 'batch_size') and config.batch_size is not None:
            command.extend(["--batch_size", str(config.batch_size)])
            
        # NOTE: min_length, max_length are dataset filtering options, not sampling script arguments
        # NOTE: output_format is handled by the pipeline, not the sampling script
            
        if hasattr(config, 'save_probabilities') and config.save_probabilities is not None:
            if config.save_probabilities:
                command.append("--save_probabilities")
                
        if hasattr(config, 'detailed_json') and config.detailed_json is not None:
            if config.detailed_json:
                command.append("--detailed_json")
        
        # Add global verbose flag if set
        if self.config.verbose:
            command.append("--verbose")
        
        # Add global no_source_indicator flag if set
        if getattr(self.config, 'no_source_indicator', False):
            command.append("--no_source_indicator")
            
        # Add global max_edge_dist flag if set
        if getattr(self.config, 'max_edge_dist', None) is not None:
            command.extend(["--max_edge_dist", str(self.config.max_edge_dist)])
        
        # Filter out parameters from additional_args that are already handled by structured config
        # to avoid duplication
        structured_args_to_filter = [
            "--verbose", "--no_source_indicator", "--max_edge_dist", "--max_structures", "--flow_temp", 
            "--steps", "--T", "--t_min", "--split", "--integration_method", "--batch_size",
            "--save_probabilities", "--detailed_json"
        ]
        
        filtered_additional_args = []
        i = 0
        while i < len(config.additional_args):
            arg = config.additional_args[i]
            if arg in structured_args_to_filter:
                # Skip this argument and its value (if it has one)
                if i + 1 < len(config.additional_args) and not config.additional_args[i + 1].startswith('--'):
                    i += 2  # Skip both the argument and its value
                else:
                    i += 1  # Skip just the argument (it's a flag)
            else:
                filtered_additional_args.append(arg)
                i += 1
        
        # Add any additional arguments (filtered to avoid duplication with structured config)
        command.extend(filtered_additional_args)
        
        try:
            # Run sampling
            result = self.run_command(command, "sampling", timeout=config.distributed.timeout_minutes * 60)
            
            if result.returncode == 0:
                logger.info("Sampling stage completed successfully")
                
                # Find the latest output file
                output_patterns = config.output_file_patterns if hasattr(config, 'output_file_patterns') else ["*.csv", "*.json"] if config.output_format == "both" else [f"*.{config.output_format}"]
                latest_output = None
                
                for pattern in output_patterns:
                    latest_output = self.find_latest_output_file(config.output_dir, pattern)
                    if latest_output:
                        break
                
                if latest_output:
                    self.stage_results["sampling"] = {
                        "status": "success",
                        "output_file": latest_output,
                        "output_dir": config.output_dir
                    }
                else:
                    logger.warning("No output file found after sampling")
                    self.stage_results["sampling"] = {
                        "status": "success",
                        "output_file": None,
                        "output_dir": config.output_dir
                    }
                
                # Copy key result files to main output directory
                self.copy_key_results_to_output_dir("sampling", config.output_dir)
                
                return True
            else:
                logger.error("Sampling stage failed")
                self.stage_results["sampling"] = {"status": "failed", "return_code": result.returncode}
                return False
                
        except Exception as e:
            logger.error(f"Sampling stage crashed: {e}")
            self.stage_results["sampling"] = {"status": "crashed", "error": str(e)}
            return False
    
    def run_evaluation_stage(self) -> bool:
        """Run the evaluation stage."""
        if not self.config.evaluation.enabled:
            logger.info("Evaluation stage disabled, skipping...")
            return True
        
        self.current_stage = "evaluation"
        logger.info("=" * 60)
        logger.info("STARTING EVALUATION STAGE")
        logger.info("=" * 60)
        
        config = self.config.evaluation
        
        # Validate reference structures directory early (fail-fast)
        if config.compare_structures and config.reference_structures_dir:
            if not os.path.exists(config.reference_structures_dir):
                logger.error(f"Reference structures directory not found: {config.reference_structures_dir}")
                return False
            if not os.path.isdir(config.reference_structures_dir):
                logger.error(f"Reference structures path is not a directory: {config.reference_structures_dir}")
                return False
            # Check if directory contains any PDB files
            pdb_files = [f for f in os.listdir(config.reference_structures_dir) if f.endswith('.pdb')]
            if not pdb_files:
                logger.error(f"No PDB files found in reference structures directory: {config.reference_structures_dir}")
                return False
            logger.info(f"Reference structures validated: {len(pdb_files)} PDB files found")
        
        # Determine predictions CSV to use
        predictions_csv = config.predictions_csv
        if predictions_csv == "auto":
            if "sampling" in self.stage_results and self.stage_results["sampling"]["status"] == "success":
                predictions_csv = self.stage_results["sampling"]["output_file"]
                if not predictions_csv:
                    logger.error("Sampling completed but no output file found")
                    return False
            else:
                logger.error("Auto predictions CSV requested but sampling did not complete successfully")
                return False
        
        if not predictions_csv or not os.path.exists(predictions_csv):
            logger.error(f"Predictions CSV not found: {predictions_csv}")
            return False
        
        # Prepare evaluation command
        if config.distributed.enabled:
            command = [
                "torchrun",
                f"--nproc_per_node={config.distributed.num_gpus}",
                f"--master_port={config.distributed.master_port}",
                "eval/evaluation_pipeline.py"
            ]
        else:
            command = ["python", "eval/evaluation_pipeline.py"]
        
        # Add evaluation arguments
        command.extend([
            "--csv_path", predictions_csv,
            "--reference_dir", config.reference_structures_dir,
            "--output-dir", config.output_dir,
            "--batch-size", str(config.batch_size),
            "--max-workers", str(config.max_workers),
            "--device", config.esmfold_device,
        ])
        
        # Add protein subset and timeout if specified
        if config.protein_subset_path:
            command.extend(["--protein-subset", config.protein_subset_path])
        if config.protein_timeout_minutes != 3.0:  # Only add if not default
            command.extend(["--protein-timeout", str(config.protein_timeout_minutes)])
        
        # Add distributed evaluation arguments
        if config.distributed.enabled:
            command.append("--distributed")
        
        # Add mode arguments
        if config.predict_structures and not config.compare_structures:
            command.append("--predict-only")
        elif config.compare_structures and not config.predict_structures:
            command.append("--compare-only")
        
        # Add overwrite flag
        if config.overwrite_predictions:
            command.append("--overwrite")
        
        # Add verbose flag
        if self.config.verbose:
            command.append("--verbose")
        
        # Filter out --verbose from additional_args to avoid duplication
        filtered_additional_args = []
        for arg in config.additional_args:
            if arg != "--verbose":
                filtered_additional_args.append(arg)
        
        # Add any additional arguments
        command.extend(filtered_additional_args)
        
        try:
            # Run evaluation
            result = self.run_command(command, "evaluation", timeout=config.distributed.timeout_minutes * 60)
            
            if result.returncode == 0:
                logger.info("Evaluation stage completed successfully")
                
                # Find the comparison results file
                results_file = self.find_latest_output_file(config.output_dir, config.comparison_results_filename)
                if not results_file:
                    results_file = self.find_latest_output_file(config.output_dir, config.output_file_patterns[0] if config.output_file_patterns else "*.csv")
                
                self.stage_results["evaluation"] = {
                    "status": "success",
                    "results_file": results_file,
                    "output_dir": config.output_dir
                }
                
                # Copy key result files to main output directory
                self.copy_key_results_to_output_dir("evaluation", config.output_dir)
                
                return True
            else:
                logger.error("Evaluation stage failed")
                self.stage_results["evaluation"] = {"status": "failed", "return_code": result.returncode}
                return False
                
        except Exception as e:
            logger.error(f"Evaluation stage crashed: {e}")
            self.stage_results["evaluation"] = {"status": "crashed", "error": str(e)}
            return False
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline."""
        logger.info("=" * 80)
        logger.info(f"STARTING PIPELINE: {self.config.name}")
        logger.info(f"Description: {self.config.description}")
        logger.info("=" * 80)
        
        success = True
        
        # Run training stage
        if self.config.training.enabled:
            if not self.run_training_stage():
                success = False
                if not self.config.retry_on_failure:
                    logger.error("Training failed, stopping pipeline")
                    return False
        
        # Run sampling stage
        if self.config.sampling.enabled:
            if not self.run_sampling_stage():
                success = False
                if not self.config.retry_on_failure:
                    logger.error("Sampling failed, stopping pipeline")
                    return False
        
        # Run evaluation stage
        if self.config.evaluation.enabled:
            if not self.run_evaluation_stage():
                success = False
                if not self.config.retry_on_failure:
                    logger.error("Evaluation failed, stopping pipeline")
                    return False
        
        # Generate final report
        self.generate_final_report()
        
        return success
    
    def generate_final_report(self):
        """Generate a final pipeline report."""
        elapsed_time = time.time() - self.start_time
        
        report = {
            "pipeline_name": self.config.name,
            "description": self.config.description,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_human": str(datetime.fromtimestamp(elapsed_time) - datetime.fromtimestamp(0)).split('.')[0],
            "stages": self.stage_results,
            "overall_success": all(
                result.get("status") == "success" 
                for result in self.stage_results.values()
            )
        }
        
        # Save report if possible
        report_path = os.path.join(self.config.base_output_dir, self.config.pipeline_report_filename)
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        except (PermissionError, FileNotFoundError):
            logger.warning(f"Cannot write report file to {report_path} - report not saved")
        
        # Print summary
        logger.info("=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Pipeline: {self.config.name}")
        logger.info(f"Total time: {report['elapsed_time_human']}")
        logger.info(f"Overall success: {report['overall_success']}")
        logger.info("")
        
        for stage_name, result in self.stage_results.items():
            status = result.get("status", "unknown")
            logger.info(f"{stage_name:12} {status.upper()}")
            
            # Print key outputs
            if status == "success":
                if stage_name == "training" and result.get("checkpoint"):
                    logger.info(f"             -> Checkpoint: {result['checkpoint']}")
                elif stage_name == "sampling" and result.get("output_file"):
                    logger.info(f"             -> Output: {result['output_file']}")
                elif stage_name == "evaluation" and result.get("results_file"):
                    logger.info(f"             -> Results: {result['results_file']}")
        
        logger.info("")
        logger.info(f"Full report: {report_path}")
        logger.info("=" * 80)

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Distributed Pipeline Orchestrator for Training, Sampling, and Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default pipeline
  python pipeline_orchestrator.py

  # Run with specific configuration
  python pipeline_orchestrator.py --config quick_test

  # Run with configuration file
  python pipeline_orchestrator.py --config-file configs/my_config.json

  # Run with YAML configuration
  python pipeline_orchestrator.py --config-file configs/production.yaml

  # Run with custom output directory
  python pipeline_orchestrator.py --output-dir /path/to/output

  # Run only sampling and evaluation
  python pipeline_orchestrator.py --config sampling_only

  # Run scaling experiments with train-loss-based LR scheduling
  python pipeline_orchestrator.py --config scaling_experiment

  # Run with modified parameters and save config
  python pipeline_orchestrator.py --config default --training-epochs 50 --sampling-num-samples 200 --save-config my_config.json

  # Run with AF2 training
  python pipeline_orchestrator.py --config-file configs/my_config.json --af2_chunk_dir $$AMLT_DATA_DIR/datasets/af2_chunks/

  # Create example configuration files
  python config_loader.py
        """
    )
    
    parser.add_argument("--config", default="default", 
                       help="Configuration to use (default, quick_test, full_scale, sampling_only, evaluation_only, scaling_experiment) or path to config file")
    parser.add_argument("--config-file", help="Path to configuration file (JSON, YAML, or TOML)")
    parser.add_argument("--save-config", help="Save current configuration to file")
    parser.add_argument("--output-dir", help="Base output directory")
    parser.add_argument("--af2_chunk_dir", help="Path to AF2 chunk directory for AF2 training (e.g., $$AMLT_DATA_DIR/datasets/af2_chunks/)")
    parser.add_argument("--af2_chunk_limit", type=int, help="Maximum number of AF2 chunks to load")
    # NOTE: AF2 data is ALWAYS loaded lazily - no upfront loading option
    parser.add_argument("--heterogeneous_batches", type=lambda x: x.lower() == 'true', help="Create heterogeneous batches mixing AF2 and PDB")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no_source_indicator", action="store_true", help="Disable data source indicators - use [1.0, 1.0] for all data sources")
    parser.add_argument("--max_edge_dist", type=float, help="Maximum distance cutoff (Angstroms) for edge creation. Overrides k_neighbors, k_farthest, k_random. Max 80 neighbors per node.")
    
    # Stage control
    parser.add_argument("--disable-training", action="store_true", help="Disable training stage")
    parser.add_argument("--disable-sampling", action="store_true", help="Disable sampling stage")
    parser.add_argument("--disable-evaluation", action="store_true", help="Disable evaluation stage")
    
    # Override parameters
    parser.add_argument("--training-epochs", type=int, help="Override training epochs")
    parser.add_argument("--sampling-num-samples", type=int, help="Override number of samples")
    parser.add_argument("--global-num-gpus", type=int, help="Override number of GPUs for all stages")
    parser.add_argument("--external-model-checkpoint", help="Path to external model checkpoint for sampling (supports AMLT $$ variables)")
    parser.add_argument("--protein-subset", help="Path to .npy file containing specific proteins to evaluate (speeds up evaluation)")
    parser.add_argument("--protein-timeout", type=float, default=3.0, help="Timeout per protein in minutes for evaluation (default: 3.0)")
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader()
    
    logger.info("Loading configuration...")
    if args.config_file:
        # Load from file
        try:
            config = config_loader.load_config(args.config_file)
            logger.info(f"Loaded configuration from file: {args.config_file}")
        except ConfigurationError as e:
            logger.error(f"CRITICAL: Failed to load configuration file: {e}")
            logger.error(f"File path: {args.config_file}")
            return 1
        except Exception as e:
            logger.error(f"CRITICAL: Unexpected error loading config file: {e}")
            logger.error(f"File path: {args.config_file}")
            raise RuntimeError(f"Configuration loading failed unexpectedly: {e}") from e
    else:
        # Load from predefined configs
        if args.config not in CONFIGS:
            # Check if it's a file path
            if os.path.exists(args.config):
                try:
                    config = config_loader.load_config(args.config)
                    logger.info(f"Loaded configuration from file: {args.config}")
                except ConfigurationError as e:
                    logger.error(f"CRITICAL: Failed to load configuration file: {e}")
                    logger.error(f"File path: {args.config}")
                    return 1
                except Exception as e:
                    logger.error(f"CRITICAL: Unexpected error loading config: {e}")
                    raise RuntimeError(f"Configuration loading failed: {e}") from e
            else:
                logger.error(f"CRITICAL: Unknown configuration: {args.config}")
                logger.error(f"Available configurations: {list(CONFIGS.keys())}")
                logger.error(f"Or specify a valid file path")
                return 1
        else:
            try:
                config = CONFIGS[args.config]
                logger.info(f"Loaded predefined configuration: {args.config}")
            except KeyError as e:
                logger.error(f"CRITICAL: Configuration {args.config} not found in CONFIGS")
                logger.error(f"Available: {list(CONFIGS.keys())}")
                raise RuntimeError(f"Predefined config missing: {e}") from e
            except Exception as e:
                logger.error(f"CRITICAL: Error accessing predefined config: {e}")
                raise RuntimeError(f"Config access failed: {e}") from e
    
    # Validate loaded configuration structure
    logger.info("Validating configuration structure...")
    try:
        if not hasattr(config, 'training'):
            raise AttributeError("Configuration missing required 'training' section")
        if not hasattr(config.training, 'additional_args'):
            raise AttributeError("Training configuration missing 'additional_args'")
        if not isinstance(config.training.additional_args, list):
            raise TypeError(f"training.additional_args must be list, got {type(config.training.additional_args)}")
        if not hasattr(config, 'base_output_dir'):
            raise AttributeError("Configuration missing 'base_output_dir'")
        logger.info("Configuration structure validation passed")
    except Exception as e:
        logger.error(f"CRITICAL: Configuration validation failed: {e}")
        logger.error(f"Config type: {type(config)}")
        logger.error(f"Config attributes: {dir(config) if hasattr(config, '__dict__') else 'No attributes'}")
        raise RuntimeError(f"Invalid configuration structure: {e}") from e
    
    # Apply overrides
    if args.output_dir:
        config.base_output_dir = args.output_dir
    
    if args.verbose:
        config.verbose = True
    
    if args.no_source_indicator:
        config.no_source_indicator = True
    
    # Graph building parameter override
    if args.max_edge_dist is not None:
        config.max_edge_dist = args.max_edge_dist
        logger.info(f"Using distance-based edge building: max_edge_dist={args.max_edge_dist}Å")
    
    # Stage control
    if args.disable_training:
        config.training.enabled = False
    if args.disable_sampling:
        config.sampling.enabled = False
    if args.disable_evaluation:
        config.evaluation.enabled = False
    
    # Parameter overrides
    if args.training_epochs:
        config.training.epochs = args.training_epochs
    if args.sampling_num_samples:
        config.sampling.num_samples = args.sampling_num_samples
    if args.global_num_gpus:
        config.training.distributed.num_gpus = args.global_num_gpus
        config.sampling.distributed.num_gpus = args.global_num_gpus
        config.evaluation.distributed.num_gpus = args.global_num_gpus
    
    # External model checkpoint override
    if args.external_model_checkpoint:
        config.sampling.model_checkpoint = args.external_model_checkpoint
        logger.info(f"Using external model checkpoint: {args.external_model_checkpoint}")
    
    # Protein subset evaluation override
    if args.protein_subset:
        config.evaluation.protein_subset_path = args.protein_subset
        logger.info(f"Using protein subset for evaluation: {args.protein_subset}")
    
    # Protein timeout override
    if hasattr(args, 'protein_timeout') and args.protein_timeout:
        config.evaluation.protein_timeout_minutes = args.protein_timeout
        logger.info(f"Using protein timeout: {args.protein_timeout} minutes")
    
    # AF2 chunk directory override for AF2 training
    if args.af2_chunk_dir:
        logger.info(f"Processing AF2 chunk directory argument: {args.af2_chunk_dir}")
        
        # Validate that config.training exists and has additional_args
        if not hasattr(config, 'training'):
            raise AttributeError("Configuration missing 'training' attribute. Check config loading.")
        if not hasattr(config.training, 'additional_args'):
            raise AttributeError("Configuration training missing 'additional_args' attribute. Check config structure.")
        if not isinstance(config.training.additional_args, list):
            raise TypeError(f"training.additional_args must be a list, got {type(config.training.additional_args)}")
        
        # Add --af2_chunk_dir to additional_args if not already present
        updated_additional_args = []
        af2_chunk_dir_found = False
        skip_next = False
        
        for i, arg in enumerate(config.training.additional_args):
            if skip_next:
                skip_next = False
                continue
                
            if arg == '--af2_chunk_dir':
                # Replace existing af2_chunk_dir
                updated_additional_args.extend(['--af2_chunk_dir', args.af2_chunk_dir])
                skip_next = True  # Skip the old value
                af2_chunk_dir_found = True
                logger.info(f"Replaced existing --af2_chunk_dir with {args.af2_chunk_dir}")
            else:
                updated_additional_args.append(arg)
        
        # If --af2_chunk_dir wasn't found, add it
        if not af2_chunk_dir_found:
            updated_additional_args.extend(['--af2_chunk_dir', args.af2_chunk_dir])
            logger.info(f"Added --af2_chunk_dir {args.af2_chunk_dir}")
        
        config.training.additional_args = updated_additional_args
        logger.info(f"Using AF2 chunk directory: {args.af2_chunk_dir}")
    else:
        # Check if AF2 is enabled without af2_chunk_dir - fail fast
        logger.info("No AF2 chunk directory provided, checking if AF2 mode requires it...")
        
        # Validate configuration structure before proceeding
        if not hasattr(config, 'training'):
            raise AttributeError("CRITICAL: Configuration missing 'training' attribute")
        if not hasattr(config.training, 'additional_args'):
            raise AttributeError("CRITICAL: Configuration training missing 'additional_args' attribute")
        if not isinstance(config.training.additional_args, list):
            raise TypeError(f"CRITICAL: training.additional_args must be a list, got {type(config.training.additional_args)}")
        
        # CRITICAL: Use static function to avoid 'self' reference error
        try:
            af2_enabled = is_af2_mode_enabled_static(config.training.additional_args)
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to check AF2 mode status: {e}")
            logger.error(f"This usually indicates a programming error in AF2 detection logic")
            logger.error(f"additional_args: {config.training.additional_args}")
            raise RuntimeError(f"AF2 mode detection failed - this is a bug: {e}") from e
        
        if af2_enabled:
            logger.error("CRITICAL ERROR: AF2 mode enabled but --af2_chunk_dir not provided.")
            logger.error("AF2 training requires access to AF2 chunk files.")
            logger.error("Please specify --af2_chunk_dir path for AF2 files (e.g., --af2_chunk_dir $$AMLT_DATA_DIR/datasets/af2_chunks/)")
            logger.error(f"Current additional_args: {config.training.additional_args}")
            raise ValueError("AF2 mode requires --af2_chunk_dir argument")
        else:
            logger.info("AF2 mode not detected, AF2 chunk directory not required")
    
    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.base_output_dir = os.path.join(config.base_output_dir, f"{config.name}_{timestamp}")
    
    # Save configuration if requested
    if args.save_config:
        try:
            config_loader.save_config(config, args.save_config)
            logger.info(f"Configuration saved to: {args.save_config}")
        except ConfigurationError as e:
            logger.error(f"Failed to save configuration: {e}")
            return 1
    
    try:
        # Final validation before creating orchestrator
        logger.info("Performing final validation before pipeline execution...")
        
        # Validate configuration integrity
        if not hasattr(config, 'name'):
            raise AttributeError("Configuration missing 'name' attribute")
        if not isinstance(config.base_output_dir, str):
            raise TypeError(f"base_output_dir must be string, got {type(config.base_output_dir)}")
        
        # Validate that we're not accidentally creating multiple orchestrator instances
        # (this was the source of the previous bug)
        logger.info(f"Creating single PipelineOrchestrator instance for config: {config.name}")
        
        # Run pipeline
        orchestrator = PipelineOrchestrator(config)
        
        # Verify orchestrator was created successfully
        if not hasattr(orchestrator, 'config'):
            raise RuntimeError("PipelineOrchestrator creation failed - missing config attribute")
        if not hasattr(orchestrator, 'is_af2_mode_enabled'):
            raise RuntimeError("PipelineOrchestrator creation failed - missing AF2 detection method")
        
        logger.info("PipelineOrchestrator created successfully, starting pipeline...")
        success = orchestrator.run_pipeline()
        
        logger.info(f"Pipeline completed with success={success}")
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except AttributeError as e:
        logger.error(f"CRITICAL: Configuration or object structure error: {e}")
        logger.error("This usually indicates missing required attributes in config or code")
        logger.error(f"Error type: {type(e).__name__}")
        raise RuntimeError(f"Attribute error in pipeline setup: {e}") from e
    except TypeError as e:
        logger.error(f"CRITICAL: Type mismatch error: {e}")
        logger.error("This usually indicates incorrect data types in configuration")
        logger.error(f"Error type: {type(e).__name__}")
        raise RuntimeError(f"Type error in pipeline setup: {e}") from e
    except Exception as e:
        logger.error(f"CRITICAL: Pipeline failed with unexpected error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"This is likely a programming error that needs investigation")
        
        # Log additional debugging information
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        raise RuntimeError(f"Pipeline execution failed: {e}") from e


def validate_environment():
    """
    Perform early validation of critical dependencies and environment.
    Fails fast if essential components are missing.
    """
    # Check Python packages
    missing_packages = []
    
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - will use CPU mode")
    except ImportError:
        missing_packages.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing_packages.append("transformers")
    
    try:
        import pandas
    except ImportError:
        missing_packages.append("pandas")
    
    try:
        import tmtools
    except ImportError:
        missing_packages.append("tmtools")
    
    if missing_packages:
        logger.error(f"Critical packages missing: {', '.join(missing_packages)}")
        logger.error("Install missing packages before running pipeline")
        return False
    
    # Check if evaluation modules can be imported
    eval_dir = os.path.join(os.path.dirname(__file__), 'eval')
    if not os.path.exists(eval_dir):
        logger.error(f"Evaluation directory not found: {eval_dir}")
        return False
    
    try:
        sys.path.append(eval_dir)
        from structure_predictor import BatchStructurePredictor
        from structure_comparator import StructureComparator
    except ImportError as e:
        logger.error(f"Failed to import evaluation modules: {e}")
        return False
    
    logger.info("Environment validation passed")
    return True


if __name__ == "__main__":
    # Validate environment first (fail-fast)
    if not validate_environment():
        logger.error("Environment validation failed - exiting")
        sys.exit(1)
    
    sys.exit(main())
