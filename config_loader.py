
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Configuration file loader and manager for pipeline orchestrator.
Supports JSON, YAML, and TOML configuration files.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict

# Try to import optional dependencies
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import toml
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

from pipeline_config import (
    PipelineConfig, TrainingConfig, SamplingConfig, EvaluationConfig, DistributedConfig
)

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass

class ConfigLoader:
    """Configuration file loader with support for multiple formats."""
    
    def __init__(self):
        self.supported_formats = ['.json']
        if YAML_AVAILABLE:
            self.supported_formats.extend(['.yaml', '.yml'])
        if TOML_AVAILABLE:
            self.supported_formats.append('.toml')
    
    def load_config(self, config_path: Union[str, Path]) -> PipelineConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            PipelineConfig instance
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        # Determine file format
        suffix = config_path.suffix.lower()
        if suffix not in self.supported_formats:
            raise ConfigurationError(
                f"Unsupported configuration format: {suffix}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
        try:
            # Load raw configuration data
            if suffix == '.json':
                config_data = self._load_json(config_path)
            elif suffix in ['.yaml', '.yml']:
                config_data = self._load_yaml(config_path)
            elif suffix == '.toml':
                config_data = self._load_toml(config_path)
            else:
                raise ConfigurationError(f"Unsupported format: {suffix}")
            
            # Convert to PipelineConfig
            return self._dict_to_config(config_data)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {e}")
    
    def save_config(self, config: PipelineConfig, config_path: Union[str, Path]):
        """
        Save configuration to file.
        
        Args:
            config: PipelineConfig instance to save
            config_path: Path where to save the configuration
            
        Raises:
            ConfigurationError: If file cannot be saved
        """
        config_path = Path(config_path)
        
        # Determine file format
        suffix = config_path.suffix.lower()
        if suffix not in self.supported_formats:
            raise ConfigurationError(
                f"Unsupported configuration format: {suffix}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
        try:
            # Convert config to dict
            config_data = asdict(config)
            
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            if suffix == '.json':
                self._save_json(config_data, config_path)
            elif suffix in ['.yaml', '.yml']:
                self._save_yaml(config_data, config_path)
            elif suffix == '.toml':
                self._save_toml(config_data, config_path)
            else:
                raise ConfigurationError(f"Unsupported format: {suffix}")
                
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {config_path}: {e}")
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not YAML_AVAILABLE:
            raise ConfigurationError("YAML support not available. Install PyYAML: pip install PyYAML")
        
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_toml(self, path: Path) -> Dict[str, Any]:
        """Load TOML configuration file."""
        if not TOML_AVAILABLE:
            raise ConfigurationError("TOML support not available. Install toml: pip install toml")
        
        with open(path, 'r') as f:
            return toml.load(f)
    
    def _save_json(self, data: Dict[str, Any], path: Path):
        """Save JSON configuration file."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_yaml(self, data: Dict[str, Any], path: Path):
        """Save YAML configuration file."""
        if not YAML_AVAILABLE:
            raise ConfigurationError("YAML support not available. Install PyYAML: pip install PyYAML")
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    
    def _save_toml(self, data: Dict[str, Any], path: Path):
        """Save TOML configuration file."""
        if not TOML_AVAILABLE:
            raise ConfigurationError("TOML support not available. Install toml: pip install toml")
        
        with open(path, 'w') as f:
            toml.dump(data, f)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> PipelineConfig:
        """Convert dictionary to PipelineConfig object."""
        
        # Helper function to create config objects from dicts
        def create_distributed_config(dist_data: Dict[str, Any]) -> DistributedConfig:
            return DistributedConfig(
                enabled=dist_data.get('enabled', True),
                num_gpus=dist_data.get('num_gpus', 4),
                master_port=dist_data.get('master_port', 29500),
                backend=dist_data.get('backend', 'nccl'),
                timeout_minutes=dist_data.get('timeout_minutes', 30)
            )
        
        def create_training_config(train_data: Dict[str, Any]) -> TrainingConfig:
            distributed = create_distributed_config(train_data.get('distributed', {}))
            
            return TrainingConfig(
                enabled=train_data.get('enabled', True),
                model_name=train_data.get('model_name', 'gvp'),
                dataset_path=train_data.get('dataset_path', 'data/cath_dataset.py'),
                config_path=train_data.get('config_path'),  # No default - will be None if not specified
                batch_size=train_data.get('batch_size', 32),
                epochs=train_data.get('epochs', 100),
                learning_rate=train_data.get('learning_rate', 1e-3),
                grad_clip=train_data.get('grad_clip', 10.0),
                time_integration=train_data.get('time_integration', 'film'),
                use_time_conditioning=train_data.get('use_time_conditioning', True),
                use_virtual_node=train_data.get('use_virtual_node', False),
                distributed=distributed,
                output_dir=train_data.get('output_dir', 'training_output'),
                checkpoint_dir=train_data.get('checkpoint_dir', 'checkpoints'),
                wandb_project=train_data.get('wandb_project', 'inverse-folding'),
                wandb_enabled=train_data.get('wandb_enabled', True),
                additional_args=train_data.get('additional_args', []),
                resume_from_checkpoint=train_data.get('resume_from_checkpoint'),
                val_split=train_data.get('val_split', 0.1),
                val_frequency=train_data.get('val_frequency', 5)
            )
        
        def create_sampling_config(sample_data: Dict[str, Any]) -> SamplingConfig:
            distributed = create_distributed_config(sample_data.get('distributed', {}))
            
            return SamplingConfig(
                enabled=sample_data.get('enabled', True),
                model_checkpoint=sample_data.get('model_checkpoint', 'auto'),
                dataset_path=sample_data.get('dataset_path', 'data/cath_dataset.py'),
                num_samples=sample_data.get('num_samples', sample_data.get('max_structures', 2000)),
                temperature=sample_data.get('temperature', sample_data.get('flow_temp', 1.0)),
                steps=sample_data.get('steps', 20),
                T=sample_data.get('T', 8.0),
                t_min=sample_data.get('t_min', 1.0),
                split=sample_data.get('split', 'validation'),
                integration_method=sample_data.get('integration_method', 'euler'),
                save_probabilities=sample_data.get('save_probabilities', True),
                detailed_json=sample_data.get('detailed_json', False),
                auto_config=sample_data.get('auto_config', True),
                distributed=distributed,
                batch_size=sample_data.get('batch_size', 16),
                max_workers=sample_data.get('max_workers', 4),
                output_dir=sample_data.get('output_dir', 'sampling_output'),
                output_format=sample_data.get('output_format', 'csv'),
                min_length=sample_data.get('min_length', 50),
                max_length=sample_data.get('max_length', 500),
                additional_args=sample_data.get('additional_args', [])
            )
        
        def create_evaluation_config(eval_data: Dict[str, Any]) -> EvaluationConfig:
            distributed = create_distributed_config(eval_data.get('distributed', {}))
            
            return EvaluationConfig(
                enabled=eval_data.get('enabled', True),
                predictions_csv=eval_data.get('predictions_csv', 'auto'),
                reference_structures_dir=eval_data.get('reference_structures_dir', 
                    'datasets/esmfold_predictions/esmfold_predictions_on_ref_valid'),
                predict_structures=eval_data.get('predict_structures', True),
                compare_structures=eval_data.get('compare_structures', True),
                distributed=distributed,
                batch_size=eval_data.get('batch_size', 8),
                max_workers=eval_data.get('max_workers', 4),
                esmfold_device=eval_data.get('esmfold_device', 'auto'),
                overwrite_predictions=eval_data.get('overwrite_predictions', False),
                output_dir=eval_data.get('output_dir', 'evaluation_output'),
                additional_args=eval_data.get('additional_args', [])
            )
        
        def create_dataset_config(dataset_data: Dict[str, Any]) -> 'DatasetConfig':
            """Create DatasetConfig from dict."""
            from pipeline_config import DatasetConfig
            return DatasetConfig(
                split_json=dataset_data.get('split_json', 'datasets/cath-4.2/chain_set_splits.json'),
                map_pkl=dataset_data.get('map_pkl', 'datasets/cath-4.2/chain_set_map_with_b_factors.pkl'),
                base_dataset_dir=dataset_data.get('base_dataset_dir', 'datasets/cath-4.2')
            )

        # Create main config object
        training_config = create_training_config(data.get('training', {}))
        sampling_config = create_sampling_config(data.get('sampling', {}))
        evaluation_config = create_evaluation_config(data.get('evaluation', {}))
        
        # Handle dataset config - look for it in multiple places with priority order:
        # 1. Global dataset section
        # 2. Training dataset section 
        # 3. Default values
        dataset_config = None
        if 'dataset' in data:
            dataset_config = create_dataset_config(data['dataset'])
        elif 'training' in data and 'dataset' in data['training']:
            dataset_config = create_dataset_config(data['training']['dataset'])
        else:
            # Use defaults
            dataset_config = create_dataset_config({})
        
        # Handle global distributed config
        global_distributed = None
        if 'global_distributed' in data and data['global_distributed'] is not None:
            global_distributed = create_distributed_config(data['global_distributed'])
        
        return PipelineConfig(
            name=data.get('name', 'pipeline'),
            description=data.get('description', 'Pipeline configuration'),
            base_output_dir=data.get('base_output_dir', 'pipeline_output'),
            verbose=data.get('verbose', True),
            cleanup_intermediate=data.get('cleanup_intermediate', False),
            dataset=dataset_config,
            training=training_config,
            sampling=sampling_config,
            evaluation=evaluation_config,
            global_distributed=global_distributed,
            conda_env=data.get('conda_env'),
            docker_image=data.get('docker_image'),
            singularity_image=data.get('singularity_image'),
            memory_limit=data.get('memory_limit'),
            time_limit=data.get('time_limit'),
            retry_on_failure=data.get('retry_on_failure', True),
            max_retries=data.get('max_retries', 3)
        )

def create_example_configs():
    """Create example configuration files in different formats."""
    
    # Create configs directory
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    loader = ConfigLoader()
    
    # Development configuration
    dev_config = PipelineConfig(
        name="development",
        description="Development configuration for quick testing",
        base_output_dir="dev_output",
        verbose=True,
        
        training=TrainingConfig(
            enabled=True,
            epochs=10,
            batch_size=16,
            learning_rate=1e-3,
            distributed=DistributedConfig(enabled=True, num_gpus=2),
            wandb_enabled=False
        ),
        
        sampling=SamplingConfig(
            enabled=True,
            num_samples=50,
            batch_size=8,
            temperature=1.0,
            distributed=DistributedConfig(enabled=True, num_gpus=2)
        ),
        
        evaluation=EvaluationConfig(
            enabled=True,
            batch_size=4,
            distributed=DistributedConfig(enabled=True, num_gpus=2),
            overwrite_predictions=True
        )
    )
    
    # Production configuration
    prod_config = PipelineConfig(
        name="production",
        description="Production configuration for full training",
        base_output_dir="production_output",
        verbose=True,
        
        training=TrainingConfig(
            enabled=True,
            epochs=500,
            batch_size=64,
            learning_rate=5e-4,
            distributed=DistributedConfig(enabled=True, num_gpus=8),
            wandb_enabled=True,
            wandb_project="inverse-folding-production"
        ),
        
        sampling=SamplingConfig(
            enabled=True,
            num_samples=2000,
            batch_size=32,
            temperature=0.8,
            distributed=DistributedConfig(enabled=True, num_gpus=8)
        ),
        
        evaluation=EvaluationConfig(
            enabled=True,
            batch_size=16,
            distributed=DistributedConfig(enabled=True, num_gpus=8)
        )
    )
    
    # Sampling-only configuration
    sampling_only_config = PipelineConfig(
        name="sampling_only",
        description="Sampling and evaluation only",
        base_output_dir="sampling_only_output",
        verbose=True,
        
        training=TrainingConfig(enabled=False),
        
        sampling=SamplingConfig(
            enabled=True,
            model_checkpoint="/path/to/pretrained/model.pt",
            num_samples=1000,
            batch_size=32,
            temperature=0.9,
            distributed=DistributedConfig(enabled=True, num_gpus=4)
        ),
        
        evaluation=EvaluationConfig(
            enabled=True,
            batch_size=16,
            distributed=DistributedConfig(enabled=True, num_gpus=4)
        )
    )
    
    # Save example configurations
    configs_to_save = [
        (dev_config, "development"),
        (prod_config, "production"),
        (sampling_only_config, "sampling_only")
    ]
    
    for config, name in configs_to_save:
        # Save in JSON format
        json_path = configs_dir / f"{name}.json"
        loader.save_config(config, json_path)
        
        # Save in YAML format if available
        if YAML_AVAILABLE:
            yaml_path = configs_dir / f"{name}.yaml"
            loader.save_config(config, yaml_path)
    
    print(f"Example configuration files created in {configs_dir}/")
    print("Available formats:", loader.supported_formats)

def main():
    """Main function to create example configurations."""
    create_example_configs()

if __name__ == "__main__":
    main()
