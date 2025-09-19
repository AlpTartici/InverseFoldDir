# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Distributed Pipeline Orchestrator Configuration

This configuration file defines all parameters for the full distributed pipeline:
- Training
- Sampling
- Evaluation

Each section can be enabled/disabled independently and contains all relevant parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class DistributedConfig:
    """Base configuration for distributed processing."""

    enabled: bool = True
    num_gpus: int = 4
    master_port: int = 29500
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    timeout_minutes: int = 30


@dataclass
class DatasetConfig:
    """Configuration for dataset paths and parameters."""

    split_json: str = "datasets/cath-4.2/chain_set_splits.json"
    map_pkl: str = (
        "datasets/cath-4.2/chain_set_map_with_b_factors_dssp.pkl"  # DSSP-enabled dataset
    )
    base_dataset_dir: str = "datasets/cath-4.2"

    def __post_init__(self):
        """Ensure paths are absolute."""
        self.split_json = os.path.abspath(self.split_json)
        self.map_pkl = os.path.abspath(self.map_pkl)
        self.base_dataset_dir = os.path.abspath(self.base_dataset_dir)


@dataclass
class TrainingConfig:
    """Configuration for distributed training."""

    enabled: bool = True

    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Model and data - use paths that match training script expectations
    model_name: str = "gvp"  # Keep for reference but not passed to training script
    dataset_path: str = (
        "data/cath_dataset.py"  # Keep for reference but not passed to training script
    )
    config_path: Optional[str] = (
        None  # This will be passed as --config_file only if specified
    )

    # Core training parameters - these should be specified explicitly in config
    batch_size: Optional[int] = None  # Must be specified
    epochs: Optional[int] = None  # Must be specified
    learning_rate: Optional[float] = None  # Must be specified
    seed: Optional[int] = None  # If None, orchestrator will generate unique seed
    grad_clip: Optional[float] = None  # Must be specified

    # Model architecture parameters - should be in additional_args for full control
    num_workers: Optional[int] = None  # Must be specified in additional_args
    dropout: Optional[float] = None  # Must be specified in additional_args
    weight_decay: Optional[float] = None  # Must be specified in additional_args
    patience: Optional[int] = None  # Must be specified in additional_args
    scheduler_patience: Optional[int] = None  # Must be specified in additional_args
    num_layers_gvp: Optional[int] = (
        None  # Must be specified in additional_args (renamed from num_layers)
    )
    num_layers_prediction: Optional[int] = (
        None  # Must be specified in additional_args (new parameter)
    )
    hidden_dim: Optional[int] = None  # Must be specified in additional_args
    hidden_dim_v: Optional[int] = None  # Must be specified in additional_args
    # architecture: str = 'blocked'  # Must be specified in additional_args: --architecture blocked|interleaved

    # Training modes and special features
    run_mode: Optional[str] = None  # Must be specified in additional_args
    overfit_protein_name: Optional[str] = None  # For overfit_on_one mode
    enable_checkpoint_rollback: Optional[bool] = (
        None  # Must be specified in additional_args
    )

    # Time conditioning parameters
    time_integration: Optional[str] = None  # Must be specified
    use_time_conditioning: Optional[bool] = None  # Must be specified
    use_virtual_node: Optional[bool] = None  # Must be specified

    # Flow matching parameters
    alpha_min: Optional[float] = None  # Must be specified in additional_args
    alpha_max: Optional[float] = None  # Must be specified in additional_args
    t_min: Optional[float] = None  # Must be specified in additional_args
    t_max: Optional[float] = None  # Must be specified in additional_args

    # DSSP multitask learning parameters
    lambda_dssp_loss: Optional[float] = (
        None  # Must be specified in additional_args (0.0 = disabled, 0.1 = 10% DSSP weight)
    )

    # Distributed training specific
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    # Output and logging
    output_dir: str = "training_output"
    checkpoint_dir: str = "checkpoints"
    checkpoint_copy_dir: Optional[str] = None  # If None, uses ../tmp by default
    wandb_project: str = "inverse-folding"
    wandb_enabled: bool = True

    # Additional training arguments - this is where most parameters should go
    # For AlphaFold2 cluster-based sampling, include:
    # "--use_af2", "--af2_cluster_dir", "datasets/af_clusters/",
    # "--af2_base_url", "https://natscidata.blob.core.windows.net/databases/raw/alphafold_database/cif/",
    # "--af2_max_retries", "5", "--af2_timeout", "30.0", "--af2_steps_per_epoch", "60"
    #
    # For scaling experiments where you want continued learning without validation-based interference:
    # "--schedule_on_train_loss"
    additional_args: List[str] = field(default_factory=list)

    # Resume training
    resume_from_checkpoint: Optional[str] = None

    # Validation (for compatibility - training script ignores these)
    val_split: float = 0.1
    val_frequency: int = 5  # Validate every N epochs

    def validate_required_fields(self):
        """Validate that all required fields are specified."""
        required_fields = [
            "batch_size",
            "epochs",
            "learning_rate",
            "grad_clip",
            "time_integration",
            "use_time_conditioning",
            "use_virtual_node",
        ]
        missing_fields = []

        for field_name in required_fields:
            if getattr(self, field_name) is None:
                missing_fields.append(field_name)

        if missing_fields:
            raise ValueError(
                f"Required training fields not specified: {missing_fields}. "
                f"All training parameters must be explicitly set in config."
            )

        # Check that essential args are in additional_args
        additional_args_str = " ".join(self.additional_args)
        recommended_args = [
            "--num_workers",
            "--dropout",
            "--weight_decay",
            "--patience",
            "--scheduler_patience",
            "--num_layers_gvp",
            "--num_layers_prediction",
            "--run_mode",
        ]
        missing_recommended = []

        for arg in recommended_args:
            if arg not in additional_args_str:
                missing_recommended.append(arg)

        if missing_recommended:
            print(
                f"Warning: Recommended args not found in additional_args: {missing_recommended}"
            )
            print("Consider adding these to additional_args for full reproducibility.")


@dataclass
class SamplingConfig:
    """Configuration for distributed sampling."""

    enabled: bool = True

    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Model and input
    model_checkpoint: str = "auto"  # "auto" to use latest from training
    dataset_path: str = "data/cath_dataset.py"

    # Sampling parameters - these come from model metadata or must be explicitly specified
    num_samples: Optional[int] = None  # Must be specified (maps to --max_structures)
    temperature: Optional[float] = None  # Must be specified (maps to --flow_temp)
    steps: Optional[int] = None  # Must be specified (maps to --steps)
    T: Optional[float] = None  # Must be specified (maps to --T)
    t_min: Optional[float] = None  # Must be specified
    split: Optional[str] = None  # Must be specified

    # Integration method - must be specified
    integration_method: Optional[str] = None  # Must be specified: "euler" or "rk45"

    # Model-dependent parameters - these MUST come from model metadata, no defaults
    # If not available in model metadata, sampling should fail
    use_virtual_node: Optional[bool] = None  # From model metadata only
    k_neighbors: Optional[int] = None  # From model metadata only
    k_farthest: Optional[int] = None  # From model metadata only
    k_random: Optional[int] = None  # From model metadata only
    max_edge_dist: Optional[float] = (
        None  # From model metadata only - distance cutoff in Angstroms
    )
    num_rbf_3d: Optional[int] = None  # From model metadata only
    num_rbf_seq: Optional[int] = None  # From model metadata only
    time_integration: Optional[str] = None  # From model metadata only
    use_time_conditioning: Optional[bool] = None  # From model metadata only

    # Output options - must be specified
    save_probabilities: Optional[bool] = None  # Must be specified
    detailed_json: Optional[bool] = None  # Must be specified
    auto_config: bool = True  # Always try to extract from model first

    # Distributed sampling specific
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    batch_size: Optional[int] = None  # Must be specified
    max_workers: Optional[int] = None  # Must be specified
    threads_per_gpu: Optional[int] = None  # Must be specified for distributed

    # Output
    output_dir: str = "sampling_output"
    output_format: Optional[str] = None  # Must be specified: csv, json, both

    # Sequence filtering - must be specified
    min_length: Optional[int] = None  # Must be specified
    max_length: Optional[int] = None  # Must be specified

    # Additional sampling arguments - for full control and reproducibility
    additional_args: List[str] = field(default_factory=list)

    def validate_required_fields(self):
        """Validate that all required fields are specified."""
        required_fields = [
            "num_samples",
            "temperature",
            "steps",
            "T",
            "t_min",
            "split",
            "integration_method",
            "save_probabilities",
            "detailed_json",
            "batch_size",
            "max_workers",
            "output_format",
            "min_length",
            "max_length",
        ]

        if self.distributed.enabled and self.threads_per_gpu is None:
            required_fields.append("threads_per_gpu")

        missing_fields = []
        for field_name in required_fields:
            if getattr(self, field_name) is None:
                missing_fields.append(field_name)

        if missing_fields:
            raise ValueError(
                f"Required sampling fields not specified: {missing_fields}. "
                f"All sampling parameters must be explicitly set in config."
            )

    def validate_model_dependent_params(self, model_metadata: dict):
        """Validate that model-dependent parameters are available from model metadata."""
        required_model_params = [
            "use_virtual_node",
            "k_neighbors",
            "k_farthest",
            "k_random",
            "max_edge_dist",
            "num_rbf_3d",
            "num_rbf_seq",
        ]
        missing_params = []

        for param in required_model_params:
            if param not in model_metadata:
                missing_params.append(param)

        if missing_params:
            raise ValueError(
                f"Required model parameters not found in model metadata: {missing_params}. "
                f"Model checkpoint must contain all necessary parameters for sampling. "
                f"Available metadata keys: {list(model_metadata.keys())}"
            )

        # Update config with model metadata
        for param in required_model_params:
            setattr(self, param, model_metadata[param])

        # Also extract optional time conditioning parameters
        optional_params = ["time_integration", "use_time_conditioning"]
        for param in optional_params:
            if param in model_metadata:
                setattr(self, param, model_metadata[param])


@dataclass
class EvaluationConfig:
    """Configuration for distributed evaluation."""

    enabled: bool = True

    # Input data
    predictions_csv: str = "auto"  # "auto" to use sampling output
    reference_structures_dir: Optional[str] = None  # Must be specified

    # Output file patterns (configurable instead of hardcoded)
    comparison_results_filename: str = "comparison_results.csv"
    output_file_patterns: List[str] = field(default_factory=lambda: ["*.csv"])

    # Evaluation modes - must be specified
    predict_structures: Optional[bool] = None  # Must be specified
    compare_structures: Optional[bool] = None  # Must be specified

    # Distributed evaluation specific
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    batch_size: Optional[int] = None  # Must be specified
    max_workers: Optional[int] = None  # Must be specified

    # ESMFold prediction
    esmfold_device: str = "auto"
    overwrite_predictions: Optional[bool] = None  # Must be specified

    # Protein subset evaluation (optional)
    protein_subset_path: Optional[str] = (
        None  # Path to .npy file with specific proteins to evaluate
    )
    protein_timeout_minutes: float = 3.0  # Timeout per protein in minutes

    # Output
    output_dir: str = "evaluation_output"

    # Additional evaluation arguments
    additional_args: List[str] = field(default_factory=list)

    def validate_required_fields(self):
        """Validate that all required fields are specified."""
        required_fields = [
            "reference_structures_dir",
            "predict_structures",
            "compare_structures",
            "batch_size",
            "max_workers",
            "overwrite_predictions",
        ]
        missing_fields = []

        for field_name in required_fields:
            if getattr(self, field_name) is None:
                missing_fields.append(field_name)

        if missing_fields:
            raise ValueError(
                f"Required evaluation fields not specified: {missing_fields}. "
                f"All evaluation parameters must be explicitly set in config."
            )


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    # Pipeline control
    name: str = "distributed_pipeline"
    description: str = (
        "Complete distributed training, sampling, and evaluation pipeline"
    )

    # Global settings
    base_output_dir: str = "pipeline_output"
    verbose: bool = True
    cleanup_intermediate: bool = False

    # Global dataset configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Configurable paths and filenames (no more hardcoded values)
    tmp_checkpoint_dir: str = "../tmp"  # Configurable tmp directory
    pipeline_report_filename: str = "pipeline_report.json"
    pipeline_config_save_filename: str = "pipeline_config.json"

    # Pipeline stages
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Global distributed settings (overrides individual stage settings if set)
    global_distributed: Optional[DistributedConfig] = None

    # Environment and containerization
    conda_env: Optional[str] = None
    docker_image: Optional[str] = None
    singularity_image: Optional[str] = None

    # Resource management
    memory_limit: Optional[str] = None  # e.g., "32GB"
    time_limit: Optional[str] = None  # e.g., "24:00:00"

    # Failure handling
    retry_on_failure: bool = True
    max_retries: int = 3

    # Validation settings
    strict_validation: bool = True  # Fail if required parameters are missing
    require_all_params_explicit: bool = True  # All parameters must be in config
    _skip_validation: bool = (
        False  # Internal flag to skip validation for default configs
    )

    def validate_configuration(self):
        """Validate the entire pipeline configuration."""
        if not self.strict_validation:
            return

        print("Validating pipeline configuration with strict mode...")

        # Validate training config
        if self.training.enabled:
            print("Validating training configuration...")
            self.training.validate_required_fields()
            print("✓ Training configuration valid")

        # Validate sampling config
        if self.sampling.enabled:
            print("Validating sampling configuration...")
            self.sampling.validate_required_fields()
            print("✓ Sampling configuration valid")

        # Validate evaluation config
        if self.evaluation.enabled:
            print("Validating evaluation configuration...")
            self.evaluation.validate_required_fields()
            print("✓ Evaluation configuration valid")

        print("✓ All pipeline configuration validation passed")

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Apply global distributed settings if specified
        if self.global_distributed:
            self.training.distributed = self.global_distributed
            self.sampling.distributed = self.global_distributed
            self.evaluation.distributed = self.global_distributed

        # Ensure output directories are absolute paths
        self.base_output_dir = os.path.abspath(self.base_output_dir)
        self.tmp_checkpoint_dir = os.path.abspath(self.tmp_checkpoint_dir)

        # Set up stage-specific output directories
        self.training.output_dir = os.path.join(self.base_output_dir, "training")
        self.sampling.output_dir = os.path.join(self.base_output_dir, "sampling")
        self.evaluation.output_dir = os.path.join(self.base_output_dir, "evaluation")

        # Set up checkpoint directory
        self.training.checkpoint_dir = os.path.join(
            self.training.output_dir, "checkpoints"
        )

        # Set checkpoint copy dir if not specified
        if self.training.checkpoint_copy_dir is None:
            self.training.checkpoint_copy_dir = self.tmp_checkpoint_dir

        # Only validate configuration if strict mode is enabled AND this is not a default config
        if (
            hasattr(self, "strict_validation")
            and self.strict_validation
            and not hasattr(self, "_skip_validation")
        ):
            self.validate_configuration()


# Default configuration
DEFAULT_CONFIG = PipelineConfig(
    name="default",
    description="Default pipeline configuration",
    strict_validation=False,  # Don't validate default configs
    training=TrainingConfig(enabled=True),
    sampling=SamplingConfig(enabled=True),
    evaluation=EvaluationConfig(enabled=True),
)

# Example configurations for different use cases
QUICK_TEST_CONFIG = PipelineConfig(
    name="quick_test",
    description="Quick test configuration with minimal resources",
    strict_validation=False,
    training=TrainingConfig(
        enabled=True,
        epochs=5,
        batch_size=8,
        learning_rate=1e-3,
        grad_clip=10.0,
        time_integration="film",
        use_time_conditioning=True,
        use_virtual_node=False,
        distributed=DistributedConfig(num_gpus=2),
    ),
    sampling=SamplingConfig(
        enabled=True,
        num_samples=20,
        batch_size=4,
        temperature=1.0,
        steps=20,
        T=8.0,
        t_min=0.0,
        split="validation",
        integration_method="euler",
        save_probabilities=True,
        detailed_json=False,
        max_workers=2,
        threads_per_gpu=1,
        output_format="csv",
        min_length=50,
        max_length=500,
        distributed=DistributedConfig(num_gpus=2),
    ),
    evaluation=EvaluationConfig(
        enabled=True,
        batch_size=4,
        max_workers=2,
        reference_structures_dir="datasets/esmfold_predictions/esmfold_predictions_on_ref_valid",
        predict_structures=True,
        compare_structures=True,
        overwrite_predictions=False,
        protein_subset_path="datasets/usable_val_proteins_under_250_for_struct_eval.npy",  # Example: evaluate only specific proteins
        protein_timeout_minutes=3.0,
        distributed=DistributedConfig(num_gpus=2),
    ),
)

FULL_SCALE_CONFIG = PipelineConfig(
    name="full_scale",
    description="Full scale training and evaluation",
    strict_validation=False,
    training=TrainingConfig(
        enabled=True,
        epochs=200,
        batch_size=64,
        learning_rate=1e-3,
        grad_clip=10.0,
        time_integration="film",
        use_time_conditioning=True,
        use_virtual_node=False,
        distributed=DistributedConfig(num_gpus=8),
    ),
    sampling=SamplingConfig(
        enabled=True,
        num_samples=1000,
        batch_size=32,
        temperature=1.0,
        steps=50,
        T=8.0,
        t_min=0.0,
        split="validation",
        integration_method="euler",
        save_probabilities=True,
        detailed_json=False,
        max_workers=4,
        threads_per_gpu=2,
        output_format="csv",
        min_length=50,
        max_length=500,
        distributed=DistributedConfig(num_gpus=8),
    ),
    evaluation=EvaluationConfig(
        enabled=True,
        batch_size=16,
        max_workers=4,
        reference_structures_dir="datasets/esmfold_predictions/esmfold_predictions_on_ref_valid",
        predict_structures=True,
        compare_structures=True,
        overwrite_predictions=False,
        distributed=DistributedConfig(num_gpus=8),
    ),
)

SAMPLING_ONLY_CONFIG = PipelineConfig(
    name="sampling_only",
    description="Only run sampling and evaluation",
    strict_validation=False,
    training=TrainingConfig(enabled=False),
    sampling=SamplingConfig(
        enabled=True,
        model_checkpoint="path/to/pretrained/model.pt",
        num_samples=500,
        temperature=1.0,
        steps=50,
        T=8.0,
        t_min=0.0,
        split="validation",
        integration_method="euler",
        save_probabilities=True,
        detailed_json=False,
        batch_size=16,
        max_workers=4,
        threads_per_gpu=2,
        output_format="csv",
        min_length=50,
        max_length=500,
        distributed=DistributedConfig(num_gpus=4),
    ),
    evaluation=EvaluationConfig(
        enabled=True,
        batch_size=8,
        max_workers=4,
        reference_structures_dir="datasets/esmfold_predictions/esmfold_predictions_on_ref_valid",
        predict_structures=True,
        compare_structures=True,
        overwrite_predictions=False,
        distributed=DistributedConfig(num_gpus=4),
    ),
)

EVALUATION_ONLY_CONFIG = PipelineConfig(
    name="evaluation_only",
    description="Only run evaluation on existing predictions",
    strict_validation=False,
    training=TrainingConfig(enabled=False),
    sampling=SamplingConfig(enabled=False),
    evaluation=EvaluationConfig(
        enabled=True,
        predictions_csv="path/to/existing/predictions.csv",
        batch_size=8,
        max_workers=4,
        reference_structures_dir="datasets/esmfold_predictions/esmfold_predictions_on_ref_valid",
        predict_structures=True,
        compare_structures=True,
        overwrite_predictions=False,
        distributed=DistributedConfig(num_gpus=4),
    ),
)

SCALING_EXPERIMENT_CONFIG = PipelineConfig(
    name="scaling_experiment",
    description="Scaling experiment configuration with train-loss-based learning rate scheduling",
    strict_validation=False,
    training=TrainingConfig(
        enabled=True,
        epochs=200,
        batch_size=64,
        learning_rate=1e-3,
        grad_clip=10.0,
        time_integration="film",
        use_time_conditioning=True,
        use_virtual_node=False,
        distributed=DistributedConfig(num_gpus=8),
        additional_args=[
            "--schedule_on_train_loss",  # Enable train-loss-based scheduling for scaling experiments
            "--num_workers",
            "8",
            "--dropout",
            "0.1",
            "--weight_decay",
            "1e-4",
            "--patience",
            "50",
            "--scheduler_patience",
            "10",
            "--num_layers_gvp",
            "3",
            "--num_layers_prediction",
            "2",
            "--hidden_dim",
            "256",
            "--hidden_dim_v",
            "64",
            "--run_mode",
            "normal",
        ],
    ),
    sampling=SamplingConfig(enabled=False),  # Disable sampling for scaling experiments
    evaluation=EvaluationConfig(
        enabled=False
    ),  # Disable evaluation for scaling experiments
)

# Configuration registry
CONFIGS = {
    "default": DEFAULT_CONFIG,
    "quick_test": QUICK_TEST_CONFIG,
    "full_scale": FULL_SCALE_CONFIG,
    "sampling_only": SAMPLING_ONLY_CONFIG,
    "evaluation_only": EVALUATION_ONLY_CONFIG,
    "scaling_experiment": SCALING_EXPERIMENT_CONFIG,
}
