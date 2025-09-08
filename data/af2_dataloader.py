"""
af2_dataloader.py

Custom DataLoader for AF2 cluster-based sampling with robust error handling.
Integrates AF2ClusterBatchSampler with AF2Dataset and handles failures gracefully.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Dict, Any
import traceback
import time

from .af2_sampler import AF2ClusterBatchSampler
from .af2_dataset import AF2Dataset
from training.collate import collate_fn


class AF2DataLoader:
    """
    Custom DataLoader for AF2 cluster-based sampling.
    
    Handles cluster sampling, protein loading, and error recovery with
    automatic cluster resampling on failures.
    """
    
    def __init__(self,
                 cluster_dir: str,
                 remote_data_dir: str,
                 batch_size: int,
                 num_batches_per_epoch: int,
                 max_retries: int = 5,
                 timeout: float = 30.0,
                 max_len: Optional[int] = None,
                 rank: int = 0,
                 world_size: int = 1,
                 seed: int = 42,
                 graph_builder_kwargs: Optional[Dict] = None,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 batch_timeout: float = 300.0,  # 5 minutes max per batch
                 max_recovery_attempts: int = 2):  # Reduce from hardcoded 3:
        """
        Initialize AF2 DataLoader.
        
        Args:
            cluster_dir: Directory containing cluster metadata
            remote_data_dir: Path to remote data directory containing AF2 CIF files
            batch_size: Number of proteins per batch (= number of clusters)
            num_batches_per_epoch: Number of batches per epoch
            max_retries: Maximum retry attempts for failed downloads
            timeout: Download timeout in seconds
            max_len: Maximum sequence length filter
            rank: Process rank for distributed training
            world_size: Total number of processes
            seed: Base random seed
            graph_builder_kwargs: Arguments for GraphBuilder
            num_workers: Number of DataLoader workers
            pin_memory: Whether to pin memory
        """
        print(f"DEBUG AF2DataLoader: Starting initialization with cluster_dir={cluster_dir}")
        print(f"DEBUG AF2DataLoader: remote_data_dir={remote_data_dir}")
        print(f"DEBUG AF2DataLoader: batch_size={batch_size}, num_batches={num_batches_per_epoch}")
        
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.rank = rank
        self.world_size = world_size
        self.max_retries = max_retries
        self.batch_timeout = batch_timeout
        self.max_recovery_attempts = max_recovery_attempts
        
        print(f"DEBUG AF2DataLoader: Creating batch sampler...")
        # Create batch sampler
        self.batch_sampler = AF2ClusterBatchSampler(
            cluster_dir=cluster_dir,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
            rank=rank,
            world_size=world_size,
            seed=seed
        )
        print(f"DEBUG AF2DataLoader: Batch sampler created successfully")
        
        print(f"DEBUG AF2DataLoader: Creating AF2Dataset...")
        # Create dataset
        self.dataset = AF2Dataset(
            remote_data_dir=remote_data_dir,
            max_retries=max_retries,
            timeout=timeout,
            max_len=max_len,
            graph_builder_kwargs=graph_builder_kwargs
        )
        print(f"DEBUG AF2DataLoader: AF2Dataset created successfully")
        
        # Store DataLoader parameters
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Circuit breaker for systematic failures
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3  # Abort after 3 consecutive batch failures
        
        print(f"AF2DataLoader initialized: batch_size={batch_size}, "
              f"num_batches_per_epoch={num_batches_per_epoch}, rank={rank}/{world_size}, "
              f"batch_timeout={batch_timeout}s, max_recovery_attempts={max_recovery_attempts}")
    
    def set_epoch(self, epoch: int):
        """Set epoch for distributed training synchronization."""
        self.batch_sampler.set_epoch(epoch)
    
    def _load_single_protein(self, uniprot_id: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Load a single protein with error handling.
        
        Args:
            uniprot_id: UniProt ID to load
            
        Returns:
            (data, y, mask) tuple or None if loading fails
        """
        try:
            print(f"DEBUG: Attempting to load protein: {uniprot_id}")
            result = self.dataset[uniprot_id]
            print(f"DEBUG: Successfully loaded protein: {uniprot_id}")
            return result
        except Exception as e:
            print(f"Failed to load protein {uniprot_id}: {e}")
            # Print more detailed error for debugging
            import traceback
            print(f"DEBUG: Full error trace for {uniprot_id}:")
            traceback.print_exc()
            return None
    
    def _resample_clusters(self, batch_idx: int, failed_proteins: List[str]) -> List[str]:
        """
        Resample clusters to replace failed proteins.
        
        Args:
            batch_idx: Current batch index
            failed_proteins: List of failed UniProt IDs
            
        Returns:
            List of replacement UniProt IDs
        """
        num_replacements = len(failed_proteins)
        
        try:
            # Get new cluster samples using numpy RNG for consistency
            worker_seed = self.batch_sampler._get_worker_seed(batch_idx + 10000)  # Offset for resampling
            rng = np.random.RandomState(worker_seed)
            
            # Sample replacement clusters
            available_clusters = self.batch_sampler.valid_cluster_indices
            if len(available_clusters) < num_replacements:
                print(f"Warning: Only {len(available_clusters)} clusters available for {num_replacements} replacements")
                num_replacements = len(available_clusters)
            
            replacement_cluster_indices = rng.choice(available_clusters, size=num_replacements, replace=False)
            
            replacement_proteins = []
            for cluster_id in replacement_cluster_indices:
                try:
                    member_id = self.batch_sampler._sample_member_from_cluster(cluster_id, rng)
                    replacement_proteins.append(member_id)
                except Exception as e:
                    print(f"Failed to sample replacement from cluster {cluster_id}: {e}")
                    continue
            
            return replacement_proteins
            
        except Exception as e:
            print(f"Failed to resample clusters: {e}")
            return []
    
    def _load_batch_with_recovery(self, protein_ids: List[str], batch_idx: int) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Load a batch of proteins with automatic error recovery and timeout.
        
        Args:
            protein_ids: List of UniProt IDs to load
            batch_idx: Current batch index
            
        Returns:
            List of successfully loaded (data, y, mask) tuples
        """
        batch_start_time = time.time()
        loaded_proteins = []
        failed_proteins = []
        
        print(f"DEBUG: Batch {batch_idx} - Attempting to load {len(protein_ids)} proteins: {protein_ids}")
        
        # First attempt: load all proteins
        for uniprot_id in protein_ids:
            # Check batch timeout
            if time.time() - batch_start_time > self.batch_timeout:
                print(f"Batch {batch_idx} timeout after {self.batch_timeout}s, stopping with {len(loaded_proteins)} proteins")
                break
                
            result = self._load_single_protein(uniprot_id)
            if result is not None:
                loaded_proteins.append(result)
            else:
                failed_proteins.append(uniprot_id)
        
        print(f"DEBUG: Batch {batch_idx} - First attempt: {len(loaded_proteins)} successful, {len(failed_proteins)} failed")
        
        # Recovery attempts for failed proteins (with reduced max attempts)
        recovery_attempts = 0
        
        while failed_proteins and recovery_attempts < self.max_recovery_attempts:
            # Check batch timeout
            if time.time() - batch_start_time > self.batch_timeout:
                print(f"Batch {batch_idx} timeout during recovery, stopping with {len(loaded_proteins)} proteins")
                break
                
            recovery_attempts += 1
            print(f"Recovery attempt {recovery_attempts}: {len(failed_proteins)} failed proteins")
            
            # Resample clusters for failed proteins
            replacement_proteins = self._resample_clusters(batch_idx, failed_proteins)
            
            if not replacement_proteins:
                print("No replacement proteins available, stopping recovery")
                break
            
            # Try loading replacement proteins
            new_failed = []
            for uniprot_id in replacement_proteins:
                # Check batch timeout
                if time.time() - batch_start_time > self.batch_timeout:
                    print(f"Batch {batch_idx} timeout during replacement loading")
                    break
                    
                result = self._load_single_protein(uniprot_id)
                if result is not None:
                    loaded_proteins.append(result)
                else:
                    new_failed.append(uniprot_id)
            
            failed_proteins = new_failed
        
        batch_time = time.time() - batch_start_time
        if failed_proteins:
            print(f"Batch {batch_idx}: {len(failed_proteins)} proteins still failed after recovery (took {batch_time:.1f}s)")
        
        # Ensure we have at least some proteins
        if not loaded_proteins:
            print(f"ERROR: Batch {batch_idx} - NO PROTEINS LOADED! All {len(protein_ids)} proteins failed")
            print(f"ERROR: Original proteins: {protein_ids}")
            print(f"ERROR: Data directory: {self.dataset.remote_data_dir}")
            raise RuntimeError(f"Failed to load any proteins for batch {batch_idx} after {batch_time:.1f}s")
        
        return loaded_proteins
    
    def __iter__(self):
        """
        Iterate over batches with error recovery and circuit breaker.
        
        Yields:
            Collated batch data (batched_graphs, padded_y, padded_masks)
        """
        print(f"DEBUG AF2DataLoader: Starting __iter__ for {self.num_batches_per_epoch} batches")
        
        try:
            batch_sampler_iter = iter(self.batch_sampler)
            print(f"DEBUG AF2DataLoader: Created batch_sampler iterator successfully")
        except Exception as e:
            print(f"ERROR AF2DataLoader: Failed to create batch_sampler iterator: {e}")
            raise
        
        for batch_idx, protein_ids in enumerate(batch_sampler_iter):
            print(f"DEBUG AF2DataLoader: Processing batch {batch_idx}, proteins: {protein_ids}")
            try:
                # Load batch with automatic recovery
                loaded_data = self._load_batch_with_recovery(protein_ids, batch_idx)
                
                if not loaded_data:
                    print(f"Skipping empty batch {batch_idx}")
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        raise RuntimeError(f"Circuit breaker: {self.consecutive_failures} consecutive batch failures")
                    continue
                
                # Collate loaded data
                try:
                    print(f"DEBUG AF2DataLoader: Collating batch {batch_idx} with {len(loaded_data)} proteins")
                    collated_batch = collate_fn(loaded_data)
                    self.consecutive_failures = 0  # Reset on success
                    print(f"DEBUG AF2DataLoader: Successfully yielding batch {batch_idx}")
                    yield collated_batch
                except Exception as e:
                    print(f"Failed to collate batch {batch_idx}: {e}")
                    traceback.print_exc()
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        raise RuntimeError(f"Circuit breaker: {self.consecutive_failures} consecutive collation failures")
                    continue
                    
            except Exception as e:
                print(f"Failed to process batch {batch_idx}: {e}")
                traceback.print_exc()
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    raise RuntimeError(f"Circuit breaker: {self.consecutive_failures} consecutive batch processing failures")
                continue
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return self.num_batches_per_epoch


def create_af2_dataloader(cluster_dir: str,
                         remote_data_dir: str,
                         batch_size: int,
                         num_batches_per_epoch: int,
                         rank: int = 0,
                         world_size: int = 1,
                         seed: int = 42,
                         batch_timeout: float = 300.0,
                         max_recovery_attempts: int = 2,
                         **kwargs) -> AF2DataLoader:
    """
    Factory function to create AF2DataLoader.
    
    Args:
        cluster_dir: Directory containing cluster metadata
        remote_data_dir: Path to remote data directory containing AF2 CIF files
        batch_size: Number of proteins per batch
        num_batches_per_epoch: Number of batches per epoch
        rank: Process rank
        world_size: Total processes
        seed: Random seed
        batch_timeout: Maximum time per batch (seconds)
        max_recovery_attempts: Maximum recovery attempts per batch
        **kwargs: Additional arguments
        
    Returns:
        Configured AF2DataLoader
    """
    return AF2DataLoader(
        cluster_dir=cluster_dir,
        remote_data_dir=remote_data_dir,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        rank=rank,
        world_size=world_size,
        seed=seed,
        batch_timeout=batch_timeout,
        max_recovery_attempts=max_recovery_attempts,
        **kwargs
    )
