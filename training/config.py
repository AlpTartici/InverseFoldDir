"""
config.py

Configuration management for the inverse folding training pipeline.
This module handles argument parsing and configuration setup to keep train.py clean.
"""

import argparse
import json
import os  # to handle output_dir override
import random
from datetime import datetime

import numpy as np  # noqa: F401
import torch  # noqa: F401


def create_parser():
    """
    Create and configure the argument parser for training.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    p = argparse.ArgumentParser(description="Train an inverse folding model with Dirichlet Flow Matching.")

    # Data arguments
    p.add_argument('--split_json', default='../datasets/cath-4.2/chain_set_splits.json',
                   help="Path to the JSON file defining data splits.")
    p.add_argument('--map_pkl', default='../datasets/cath-4.2/chain_set_map_with_b_factors_dssp.pkl',
                   help="Path to the pickle file mapping chain IDs to data (with pre-computed B-factors).")

    # Training hyperparameters
    p.add_argument('--batch', type=int, default=128, help="Batch size for training. Larger batches (128-256) can give 2-4x speedup on GPU.")
    p.add_argument('--val_batch', type=int, default=None, help="Batch size for validation. If None, uses same as --batch.")
    p.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    p.add_argument('--max_batches_per_epoch', type=int, default=None,
                   help="Maximum number of batches per epoch. If None, uses all available batches.")
    p.add_argument('--lr', type=float, default=1e-2, help="Learning rate for the Adam optimizer.")
    p.add_argument('--dropout', type=float, default=0.1, help="Dropout rate in the model.")
    p.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for AdamW.")
    p.add_argument('--grad_clip', type=float, default=5.0, help="Gradient clipping by norm. Set to 0 to disable.")

    # DSSP multitask learning parameter
    p.add_argument('--lambda_dssp_loss', type=float, default=0.0,
                   help="Weight for DSSP secondary structure prediction loss in multitask learning. "
                        "0.0 = sequence-only, 0.1 = 10% DSSP + 90% sequence, 1.0 = equal weighting.")

    # Position prediction parameters (simplified for position-only approach)

    p.add_argument('--time_sampling_strategy', type=str, default='exponential', choices=['exponential'],
                   help="Time sampling strategy: 'uniform' (uniform random sampling) or 'curriculum' (curriculum-guided sampling that biases time distribution).")

    p.add_argument('--patience', type=int, default=200, help="Early stopping patience.")
    p.add_argument('--scheduler_patience', type=int, default=3, help="Learning rate scheduler patience (epochs without improvement before LR reduction).")
    p.add_argument('--schedule_on_train_loss', action='store_true', help="Schedule learning rate on training loss instead of validation loss. Useful for scaling experiments where you want continued learning without validation-based interference.")
    p.add_argument('--val_metric', type=str, default='val_fixed', choices=['training', 'val_fixed', 'val_unfixed', 'val_combo', 'val_all'],
                   help="Validation metric for early stopping, learning rate scheduling, and checkpoint rollback. "
                        "'training': use training loss (equivalent to --schedule_on_train_loss), "
                        "'val_fixed': use fixed-time validation loss, "
                        "'val_unfixed': use unfixed-time validation loss, "
                        "'val_combo': use average of val_fixed and val_unfixed losses, "
                        "'val_all': use average of val_fixed, val_unfixed, and validation losses at t=2, t=4, t=6.")
    p.add_argument('--min_lr_stop_point', type=float, default=1e-11, help="Minimum learning rate threshold. Training stops when LR drops below this value.")

    # Validation parameters (for orchestrator compatibility)
    p.add_argument('--resume_from_checkpoint', type=str, default=None, help="Path to checkpoint to resume training from.")

    p.add_argument('--use_checkpoint_opt_and_sched', action='store_true', help="Use checkpoint optimizer and scheduler.")

    # Sampling-only mode (skip training)
    p.add_argument('--sampling_only', action='store_true',
                   help="Skip training and run sampling+evaluation only using external checkpoint.")
    p.add_argument('--external_checkpoint', type=str, default=None,
                   help="Path or URL to external checkpoint for sampling-only mode (supports Azure blob URLs).")

    # Logging and monitoring
    p.add_argument('--use_wandb', action='store_true', help="Enable wandb logging.")
    p.add_argument('--wandb_project', type=str, default='inverse-folding', help="Wandb project name.")

    # Model architecture arguments
    p.add_argument('--use_qkv', action='store_true', default=True, help="Use QKV attention instead of simple attention.")
    p.add_argument('--num_layers_gvp', type=int, default=None, help="Number of GVP-GNN layers (depth of the model).")
    p.add_argument('--num_message_layers', type=int, default=None, help="Number of message passing layers for graph communication (default: 1).")
    p.add_argument('--num_layers_prediction', type=int, default=2, help="Number of layers in prediction head (default: 2).")
    p.add_argument('--hidden_dim', type=int, default=None, help="Hidden dimension for scalar features in GNN.")
    p.add_argument('--hidden_dim_v', type=int, default=None, help="Hidden dimension for vector features in GNN.")
    p.add_argument('--architecture', type=str, default='interleaved', choices=['interleaved'],
                   help="Network architecture: 'blocked' (GVP then attention layers) or 'interleaved' (GVP and attention interleaved). Default: blocked for backward compatibility.")

    # Flexible loss scaling
    p.add_argument('--flexible_loss_scaling', action='store_true',
                   help="Enable flexible loss scaling based on uncertainty (B-factors/pLDDT scores). Higher confidence residues get higher loss weights.")

    # Debug mode
    p.add_argument('--debug_mode', action='store_true',
                   help="Enable debug mode with detailed logging and analysis (B-factor weights, gradient info, etc.).")

    # Hybrid training arguments (PDB + AF2)
    p.add_argument('--ratio_af2_pdb', type=int, default=0,
                   help="Hybrid training ratio: 0=all PDB, -1=all AF2, X=1 PDB per X AF2 structures")
    p.add_argument('--pdb_directory', type=str, default='../datasets/all_chain_pdbs/',
                   help="Directory containing PDB files with B-factors")
    p.add_argument('--af2_chunk_dir', type=str, required=False,
                   help="Directory containing AF2 pickle chunks (required when ratio_af2_pdb != 0)")
    p.add_argument('--af2_chunk_limit', type=int, required=False,
                   help="Maximum number of AF2 chunks to load (None = load all chunks)")
    # AF2 data is always loaded lazily - no upfront loading option
    p.add_argument('--heterogeneous_batches', type=lambda x: x.lower() == 'true', default=True,
                   help="Create heterogeneous batches mixing AF2 and PDB within each batch (default: True)")
    p.add_argument('--alternating_pure_batches', action='store_true', default=False,
                   help="Use alternating pure batches instead of mixed batches (overrides heterogeneous_batches)")
    p.add_argument('--verbose', action='store_true', default=False,
                   help="Enable verbose logging output")
    p.add_argument('--epoch_timeout', type=float, default=4200.0,
                   help="Timeout in seconds for each training epoch (default: 70 minutes = 4200 seconds)")
    p.add_argument('--max_error_count', type=int, default=10,
                   help="Maximum number of data loading errors before failing")
    p.add_argument('--fail_fast', action='store_true', default=False,
                   help="Fail immediately when max_error_count is reached")

    # Hybrid-specific error handling
    p.add_argument('--hybrid_max_pdb_errors', type=int, default=10,
                   help="Maximum number of PDB loading errors in hybrid mode")
    p.add_argument('--hybrid_max_af2_errors', type=int, default=10,
                   help="Maximum number of AF2 loading errors in hybrid mode")
    p.add_argument('--hybrid_fail_fast', action='store_true', default=False,
                   help="Fail fast when error thresholds are reached in hybrid mode")
    p.add_argument('--node_dim_s', type=int, default=None, help="Input scalar node feature dimension (includes geometry missing flag).")
    p.add_argument('--node_dim_v', type=int, default=None, help="Input vector node feature dimension.")
    p.add_argument('--edge_dim_s', type=int, default=None, help="Input scalar edge feature dimension.")
    p.add_argument('--edge_dim_v', type=int, default=None, help="Input vector edge feature dimension.")
    p.add_argument('--recycle_steps', type=int, default=None, help="Number of recycling iterations (minimal default).")
    p.add_argument('--time_dim', type=int, default=64, help="Time embedding dimension (must be even for Gaussian Fourier projection).")
    p.add_argument('--head_hidden', type=int, default=None, help="Hidden dimension for prediction head.")
    p.add_argument('--time_scale', type=float, default=1.0, help="Scale for Gaussian Fourier time embedding.")

    # Time conditioning parameters (new hybrid approach)
    p.add_argument('--disable_time_conditioning', action='store_true', default=False,
                   help="Disable time conditioning in GVP layers (default: enabled)")
    p.add_argument('--time_integration', type=str, default='film', choices=['film', 'add'],
                   help="Time integration method: film (FiLM) or add (additive)")

    # Structure noise parameters
    p.add_argument('--structure_noise_mag_std', type=float, default=0.0,
                   help="Standard deviation for Gaussian noise added to atom coordinates. Set to 0.0 to disable noise (default: 0.0).")
    p.add_argument('--time_based_struct_noise', type=str, default='fixed', choices=['increasing', 'decreasing', 'fixed'],
                   help="Time-based structure noise scaling (unused during training): 'increasing', 'decreasing', or 'fixed'. Only used for sampling if struct_noise_in_sampling is enabled.")
    p.add_argument('--uncertainty_struct_noise_scaling', action='store_true', default=False,
                   help="Scale structure noise based on uncertainty: more flexible parts get more noise (default: False)")
    p.add_argument('--struct_noise_in_sampling', action='store_true', default=False,
                   help="Apply coordinate noise during sampling (uses time_based_struct_noise setting)")

    # Flow matching parameters
    p.add_argument('--alpha_min', type=float, default=1.0, help="Minimum alpha for Dirichlet flow.")
    p.add_argument('--alpha_max', type=float, default=8.0, help="Maximum alpha for Dirichlet flow.")
    p.add_argument('--alpha_spacing', type=float, default=0.01, help="Alpha spacing for Dirichlet flow.")
    p.add_argument('--t_min', type=float, default=0.0, help="Minimum time for sampling.")
    p.add_argument('--t_max', type=float, default=8.0, help="Maximum time for sampling.")
    p.add_argument('--alpha_range', type=float, default=None, help="Range scaling factor for exponential time sampling in Dirichlet flow matching.")
    p.add_argument('--dirichlet_multiplier_training', type=float, default=1.0,
                   help="Multiplier for Dirichlet concentration parameters during training. Higher values increase confidence (less noise). Default: 1.0 (no change).")

    # System arguments
    p.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers. Reduced from 8 to 4 to minimize RBF table memory usage.")
    p.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument('--device', type=str, default='auto',
                   help="Device to use: 'auto', 'cpu', 'cuda', or specific GPU like 'cuda:0'. Default 'auto' uses GPU if available.")
    # Config file for hyperparameters and seed
    p.add_argument('--config_file', type=str, default=None,
                   help="Path to JSON config file defining hyperparameters and seed.")

    # Distributed training arguments
    p.add_argument('--distributed', action='store_true',
                   help="Enable distributed training (optional)")
    p.add_argument('--local_rank', type=int, default=0,
                   help="Local rank for distributed training (set automatically by torchrun)")

    # Training features
    p.add_argument('--enable_checkpoint_rollback', action='store_true',
                   help="Enable checkpoint rollback: restore best model state when LR drops and no improvement after half patience. Optimizer state is reset for fresh exploration.")
    p.add_argument('--rollback_patience_factor', type=float, default=0.51,
                   help="Fraction of scheduler patience to wait before rollback (default: 0.5, meaning half of scheduler patience).")

    # Run modes
    p.add_argument('--run_mode', type=str, default='regular',
                   choices=['regular', 'smoke_test', 'overfit', 'overfit_on_one'],
                   help="Run mode: 'regular' for full training, 'smoke_test' for a quick forward pass, " \
                   "'overfit' for debugging on a small batch, 'overfit_on_one' for overfitting on a single sample.")
    p.add_argument('--overfit_protein_name', type=str, default=None,
                   help="Specific protein name to overfit on (for overfit_on_one mode). If not provided, uses first validation sample.")

    # Model saving arguments
    p.add_argument('--model_name_prefix', type=str, default='dfm_model',
                   help="Prefix for saved model names (timestamp will be appended).")
    p.add_argument('--save_intermediate_models', default=True, action='store_true',
                   help="Save intermediate model checkpoints during training.")

    # Optional base output directory
    p.add_argument('--output_dir', type=str, default=None,
                   help="Base directory for outputs (default: ../output)")

    # Checkpoint copy directory (for pipeline orchestrator)
    p.add_argument('--checkpoint_copy_dir', type=str, default=None,
                   help="Directory to copy checkpoints to (used by pipeline orchestrator)")

    # Virtual node usage
    p.add_argument('--use_virtual_node', action='store_true', default=False,
                   help="Enable virtual node connectivity (optional)")

    # Graph building parameters
    p.add_argument('--k_neighbors', type=int, default=None,
                   help="Number of nearest neighbors per node in graph construction (default: 32)")
    p.add_argument('--k_farthest', type=int, default=None,
                   help="Number of farthest neighbors per node in graph construction (default: 16)")
    p.add_argument('--k_random', type=int, default=None,
                   help="Number of random neighbors per node in graph construction (default: 200)")
    p.add_argument('--max_edge_dist', type=float, default=None,
                   help="Maximum distance cutoff (Angstroms) for edge creation. If set, overrides k_neighbors, k_farthest, k_random. Max 80 neighbors per node for safety.")
    p.add_argument('--num_rbf_3d', type=int, default=None,
                   help="Number of RBF features for 3D distances in graph construction (default: 16)")
    p.add_argument('--num_rbf_seq', type=int, default=None,
                   help="Number of RBF features for sequence distances in graph construction (default: 16)")

    # RBF distance range parameters
    p.add_argument('--rbf_3d_min', type=float, default=1.0,
                   help="Minimum distance (Angstroms) for 3D RBF centers (default: 1.0)")
    p.add_argument('--rbf_3d_max', type=float, default=20.0,
                   help="Maximum distance (Angstroms) for 3D RBF centers (default: 20.0)")
    p.add_argument('--rbf_3d_spacing', type=str, default='exponential', choices=['linear', 'exponential'],
                   help="Spacing type for 3D RBF centers (default: exponential)")

    # Label smoothing arguments (updated for position prediction)
    p.add_argument('--use_smoothed_labels', action='store_true', default=False,
                   help="Enable smoothed labels for training (validation will still use hard labels).")
    # Removed: use_smoothed_velocity_targets (not applicable for position prediction)
    p.add_argument('--label_similarity_csv', type=str, default="../df_combined_for_one_hot.csv",
                   help="Path to CSV file containing amino acid similarity matrix for label smoothing.")
    p.add_argument('--penalize_x_prediction', action='store_true', default=True,
                   help="Penalize predicting X (unknown) amino acid by setting low probabilities.")
    p.add_argument('--blur_uncertainty', action='store_true', default=False,
                   help="Apply blur to B-factor input")

    # print all arguments
    # Note: args doesn't exist yet - this is just the parser creation function
    # Argument printing happens in get_training_config() after parsing

    return p


def create_validation_parser():
    """
    Create argument parser for validation script.

    Returns:
        argparse.ArgumentParser: Configured argument parser for validation
    """
    parser = argparse.ArgumentParser(description="Validate a trained model.")
    parser.add_argument('--split_json', default='../datasets/cath-4.2/chain_set_splits.json',
                       help="Path to the JSON file defining data splits.")
    parser.add_argument('--map_pkl', default='../datasets/cath-4.2/chain_set_map_with_b_factors.pkl',
                       help="Path to the pickle file mapping chain IDs to data (with pre-computed B-factors).")
    parser.add_argument('--model_path', default='../output/saved_models/best_model.pt',
                       help="Path to the trained model checkpoint.")
    parser.add_argument('--run_mode', type=str, default='regular',
                       choices=['regular', 'overfit_on_one'],
                       help="Run mode: 'regular' for full validation, 'overfit_on_one' for single sample validation.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for validation.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    return parser


def generate_model_name(prefix="dfm_model", include_config=True, args=None):
    """
    Generate a unique model name with timestamp to prevent overloading.

    Args:
        prefix (str): Prefix for the model name
        include_config (bool): Whether to include key config parameters in the name
        args: Parsed arguments object (optional, for including config info)

    Returns:
        str: Unique model name with timestamp
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Base name with timestamp
    model_name = f"{prefix}_{timestamp}"

    # Optionally include key configuration parameters
    if include_config and args is not None:
        config_parts = []

        # Add key hyperparameters
        config_parts.append(f"lr{args.lr:.0e}")
        config_parts.append(f"b{args.batch}")
        config_parts.append(f"gvp{args.num_layers_gvp}")
        config_parts.append(f"head{args.num_layers_prediction}")
        config_parts.append(f"h{args.hidden_dim}")

        # Add run mode if not regular
        if args.run_mode != 'regular':
            config_parts.append(args.run_mode)

        # Join config parts
        if config_parts:
            config_str = "_".join(config_parts)
            model_name = f"{model_name}_{config_str}"

    return model_name


def setup_output_directories(base_path="../output"):
    """
    Set up output directories for model saving and logging.

    Args:
        base_path (str): Base path for output directories

    Returns:
        str: The actual output base path used
    """
    print(f"setup_output_directories called with base_path: {base_path}")

    # Safety check: Never try to create directories in /workspace
    if "/workspace" in str(base_path):
        print(f"ERROR: Detected /workspace in path: {base_path}")
        print(f"Environment variables:")
        for key, value in os.environ.items():
            if 'OUTPUT' in key.upper() or 'AMLT' in key.upper():
                print(f"  {key}={value}")

        # Check if we have an environment variable with the correct AMLT output directory
        amlt_output = os.environ.get('AMLT_OUTPUT_DIR')
        if amlt_output:
            # Replace the /workspace part with the AMLT output directory
            # If the path is like "/workspace/output/training", we want to use "{AMLT_OUTPUT_DIR}/training"
            path_parts = base_path.split('/')
            if 'output' in path_parts:
                # Find the index of 'output' and take everything after it
                output_index = path_parts.index('output')
                remaining_parts = path_parts[output_index + 1:]  # Skip 'output' itself
                if remaining_parts:
                    output_base = os.path.join(amlt_output, *remaining_parts)
                else:
                    output_base = amlt_output
                print(f"Corrected /workspace path to AMLT output directory: {output_base}")
            else:
                output_base = amlt_output
                print(f"Using AMLT output directory instead of /workspace: {output_base}")
        else:
            raise ValueError(f"Attempting to create directories in /workspace is not allowed in containerized environments. "
                            f"Received path: {base_path}. Please ensure AMLT output directory is properly configured and "
                            f"AMLT_OUTPUT_DIR environment variable is set.")
    else:
        # If base_path is absolute (e.g., from AMLT output directory), use it directly
        if os.path.isabs(base_path):
            output_base = base_path
            print(f"Using absolute output directory: {output_base}")
        else:
            # For relative paths, use as-is (development/local environments)
            output_base = base_path
            print(f"Using relative output directory: {output_base}")

    try:
        os.makedirs(output_base, exist_ok=True)  # Ensure output directory exists
        os.makedirs(os.path.join(output_base, 'saved_models'), exist_ok=True)  # Ensure saved models directory exists
        print(f"Output directories created successfully at: {output_base}")
    except Exception as e:
        print(f"Error creating output directories: {e}")
        raise

    return output_base


def setup_reproducibility(seed=42, rank=None):
    """
    Set up reproducibility by setting random seeds.

    Args:
        seed (int): Random seed to use
        rank (int, optional): Process rank for distributed training.
                              If provided, each rank gets a unique seed offset.
    """
    if rank is not None:
        # Make each process use a different base seed to prevent identical initialization
        # and random sampling across distributed processes
        process_seed = seed + rank * 10000  # Large offset to avoid collisions
        print(f"Rank {rank}: Using process seed {process_seed} (base: {seed})")
    else:
        process_seed = seed
        print(f"Single process: Using seed {process_seed}")

    # Seed all RNGs with the process-specific seed
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    np.random.seed(process_seed)
    random.seed(process_seed)
    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_config(args):
    """
    Validate configuration arguments and apply any necessary adjustments.

    Args:
        args: Parsed arguments object

    Returns:
        args: Validated and potentially modified arguments
    """
    # Adjust patience for overfit modes
    if args.run_mode == 'overfit_on_one' or args.run_mode == 'overfit':
        args.patience = args.epochs  # Set patience to epochs for overfit mode
        print(f"Overfit mode detected: set patience to {args.patience}")

    # Validate hybrid training configuration
    if args.ratio_af2_pdb != 0:
        # AF2 data is needed for hybrid or AF2-only training
        if not args.af2_chunk_dir:
            raise ValueError("--af2_chunk_dir is required when ratio_af2_pdb != 0")

        # Validate AF2 chunk directory exists
        if not os.path.exists(args.af2_chunk_dir):
            raise ValueError(f"AF2 chunk directory does not exist: {args.af2_chunk_dir}")

        print(f"AF2 chunk directory: {args.af2_chunk_dir}")

    # Validate epoch timeout parameter
    if args.epoch_timeout <= 0:
        raise ValueError(f"epoch_timeout must be > 0, got {args.epoch_timeout}")

    if args.ratio_af2_pdb != -1:
        # PDB data is needed for PDB-only or hybrid training
        # Ensure CATH data is available
        if not args.split_json or not args.map_pkl:
            raise ValueError("CATH split_json and map_pkl are required when ratio_af2_pdb != -1")

        # Validate PDB directory if provided
        if args.pdb_directory and not os.path.exists(args.pdb_directory):
            print(f"Warning: PDB directory not found: {args.pdb_directory}")

    # Validate ratio parameter
    if args.ratio_af2_pdb < -1:
        raise ValueError(f"ratio_af2_pdb must be >= -1, got {args.ratio_af2_pdb}")

    # Validate error handling parameters
    if args.max_error_count < 1:
        raise ValueError(f"max_error_count must be >= 1, got {args.max_error_count}")

    # Log training mode
    if args.ratio_af2_pdb == 0:
        print("Training mode: PDB-only (B-factors)")
    elif args.ratio_af2_pdb == -1:
        print("Training mode: AF2-only (pLDDT scores)")
    else:
        print(f"Training mode: Hybrid (1 PDB per {args.ratio_af2_pdb} AF2 structures)")

        # Print AF2 configuration - use chunk directory for new system
        if hasattr(args, 'af2_chunk_dir') and args.af2_chunk_dir:
            print(f"AF2 configuration validated: chunk_dir={args.af2_chunk_dir}")
        elif hasattr(args, 'af2_remote_data') and args.af2_remote_data:
            # Fallback for old remote data system (if still used)
            print(f"AF2 configuration validated: remote_data={args.af2_remote_data}, "
                  f"cluster_dir={getattr(args, 'af2_cluster_dir', 'N/A')}, "
                  f"max_retries={getattr(args, 'af2_max_retries', 'N/A')}, "
                  f"timeout={getattr(args, 'af2_timeout', 'N/A')}s")
        else:
            print("AF2 configuration: Using hybrid mode with unknown AF2 backend")

    # Validate rollback parameters
    if args.enable_checkpoint_rollback:
        if not (0.0 < args.rollback_patience_factor <= 1.0):
            raise ValueError(f"rollback_patience_factor must be between 0 and 1, got {args.rollback_patience_factor}")

        rollback_patience_threshold = int(round(args.scheduler_patience * args.rollback_patience_factor))
        print(f"Checkpoint rollback enabled: will rollback after {rollback_patience_threshold} epochs without improvement post-LR reduction")

    # Handle validation metric configuration and maintain backward compatibility
    if args.schedule_on_train_loss and args.val_metric != 'training':
        print(f"Warning: --schedule_on_train_loss is set but --val_metric is '{args.val_metric}'. "
              f"Setting --val_metric to 'training' for consistency.")
        args.val_metric = 'training'

    # Log validation metric choice
    metric_descriptions = {
        'training': 'training loss (no validation-based interference)',
        'val_fixed': 'fixed-time validation loss (deterministic time sampling)',
        'val_unfixed': 'unfixed-time validation loss (random time sampling)',
        'val_combo': 'combined validation loss (average of fixed and unfixed)',
        'val_all': 'comprehensive validation loss (average of fixed, unfixed, t=2, t=4, t=6)'
    }
    print(f"Validation metric for early stopping/LR scheduling: {args.val_metric} - {metric_descriptions[args.val_metric]}")

    # Validate dimensions
    if args.time_dim % 2 != 0:
        raise ValueError(f"time_dim must be even for Gaussian Fourier projection, got {args.time_dim}")

    if args.time_dim <= 0:
        raise ValueError(f"time_dim must be positive, got {args.time_dim}")

    # Validate structure noise parameters
    if args.structure_noise_mag_std < 0:
        raise ValueError(f"structure_noise_mag_std must be non-negative, got {args.structure_noise_mag_std}")

    # Only validate noise-related options if noise is actually enabled
    if args.structure_noise_mag_std > 0:
        # Validate time-based noise options (categorical validation already handled by choices)
        print(f"Structure noise enabled: std={args.structure_noise_mag_std}, "
              f"uncertainty_scaling={args.uncertainty_struct_noise_scaling}, "
              f"sampling_noise={args.struct_noise_in_sampling}")
        if args.struct_noise_in_sampling:
            print(f"  Sampling noise will use time_based={args.time_based_struct_noise}")
    else:
        print("Structure noise disabled (std=0.0)")

    # Validate max_batches_per_epoch if specified
    if args.max_batches_per_epoch is not None and args.max_batches_per_epoch <= 0:
        raise ValueError(f"max_batches_per_epoch must be positive, got {args.max_batches_per_epoch}")

    # Validate graph building parameters (fail-fast approach)
    if not (args.max_edge_dist is None or args.max_edge_dist == 0):
        if args.max_edge_dist <= 0:
            raise ValueError(f"max_edge_dist must be positive, got {args.max_edge_dist}")

        # When max_edge_dist is set, override other k parameters to prevent conflicts
        if args.k_neighbors is not None or args.k_farthest is not None or args.k_random is not None:
            print(f"max_edge_dist={args.max_edge_dist} specified: overriding k_neighbors, k_farthest, k_random to None")
            args.k_neighbors = None
            args.k_farthest = None
            args.k_random = None

        print(f"Graph building mode: Distance-based edges (max_edge_dist={args.max_edge_dist}Å, max 80 neighbors per node)")
    else:
        print("Graph building mode: K-neighbor based edges")

    # Set val_batch to batch if not specified
    if args.val_batch is None:
        args.val_batch = args.batch
        print(f"Setting val_batch to same as batch: {args.val_batch}")

    return args


def get_training_config():
    """
    Main function to get complete training configuration.

    Returns:
        tuple: (args, output_base, model_name) - parsed arguments, output directory, and model name
    """
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    # Override args with values from config file if provided
    if args.config_file:
        with open(args.config_file, 'r') as f:
            cfg = json.load(f)
        for key, val in cfg.items():
            if hasattr(args, key):
                setattr(args, key, val)
            else:
                print(f"Warning: Unknown config key '{key}' in {args.config_file}")

    # Validate configuration
    args = validate_config(args)

    # Print all arguments for debugging
    print("\n[DEBUG] All parsed arguments:")
    for arg in vars(args):
        print(f"[DEBUG] Argument '{arg}': {getattr(args, arg)}")
    print("")  # Empty line for readability

    # Set up reproducibility
    setup_reproducibility(args.seed)

    # Determine base path for outputs (allow override via CLI)
    base = args.output_dir if args.output_dir is not None else None
    if base:
        output_base = setup_output_directories(base)
    else:
        output_base = setup_output_directories()

    # Generate unique model name
    model_name = generate_model_name(args.model_name_prefix, include_config=True, args=args)
    print(f"Generated model name: {model_name}")

    return args, output_base, model_name
