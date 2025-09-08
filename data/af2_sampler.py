"""
af2_sampler.py

AlphaFold2 cluster-based batch sampler for scalable protein training.
Samples one protein from each of batch_size clusters per iteration.
"""
import os
import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Iterator, List


class AF2ClusterBatchSampler(Sampler):
    """
    Custom batch sampler for AlphaFold2 cluster-based sampling.
    
    Samples batch_size different clusters per iteration, then randomly selects
    one member (UniProt ID) from each selected cluster. Ensures no cluster
    overlap within a batch and provides worker-specific seeding for distributed
    training.
    """
    
    def __init__(self, 
                 cluster_dir: str,
                 batch_size: int,
                 num_batches_per_epoch: int,
                 rank: int = 0,
                 world_size: int = 1,
                 seed: int = 42,
                 drop_last: bool = True):
        """
        Initialize AF2 cluster batch sampler.
        
        Args:
            cluster_dir: Directory containing cluster metadata (.npy files)
            batch_size: Number of clusters to sample per batch
            num_batches_per_epoch: Number of batches per epoch
            rank: Process rank for distributed training
            world_size: Total number of processes
            seed: Base random seed
            drop_last: Whether to drop incomplete batches
        """
        self.cluster_dir = cluster_dir
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # Enable verbose mode for small batch counts (debugging)
        self.verbose = num_batches_per_epoch < 5
        
        print(f"DEBUG AF2ClusterBatchSampler: Starting cluster metadata loading from {cluster_dir}")
        # Load cluster metadata with memory mapping for efficiency
        self._load_cluster_metadata()
        print(f"DEBUG AF2ClusterBatchSampler: Cluster metadata loaded successfully")
        
        # Validate that we have enough clusters
        if self.num_clusters < batch_size:
            raise ValueError(f"Not enough clusters ({self.num_clusters}) for batch size ({batch_size})")
        
        if self.verbose:
            print(f"AF2ClusterBatchSampler initialized (VERBOSE MODE): {self.num_clusters} clusters, "
                  f"batch_size={batch_size}, rank={rank}/{world_size}, "
                  f"num_batches_per_epoch={num_batches_per_epoch}", flush=True)
        else:
            print(f"AF2ClusterBatchSampler initialized: {self.num_clusters} clusters, "
                  f"batch_size={batch_size}, rank={rank}/{world_size}")
    
    def _load_cluster_metadata(self):
        """Load cluster metadata from memory-mapped numpy arrays."""
        cluster_sizes_path = os.path.join(self.cluster_dir, 'cluster_sizes.npy')
        cluster_offsets_path = os.path.join(self.cluster_dir, 'cluster_offsets.npy')
        flat_members_path = os.path.join(self.cluster_dir, 'flat_members.npy')
        
        print(f"DEBUG AF2ClusterBatchSampler: Checking cluster metadata files...")
        print(f"  cluster_sizes_path: {cluster_sizes_path}")
        print(f"  cluster_offsets_path: {cluster_offsets_path}")
        print(f"  flat_members_path: {flat_members_path}")
        
        # Skip file existence checks (Azure blob storage can timeout)
        for path in [cluster_sizes_path, cluster_offsets_path, flat_members_path]:
            print(f"DEBUG AF2ClusterBatchSampler: Will attempt to load: {path}")
            # Let numpy.load() handle file not found errors instead of os.path.exists()
        
        print(f"DEBUG AF2ClusterBatchSampler: Loading cluster_sizes...")
        # Memory-map arrays for efficient shared access across workers
        try:
            self.cluster_sizes = np.load(cluster_sizes_path, mmap_mode='r')
            print(f"DEBUG AF2ClusterBatchSampler: Loaded cluster_sizes, shape: {self.cluster_sizes.shape}")
        except:
            try:
                self.cluster_sizes = np.load("/workspace/datasets/af_clusters/cluster_sizes.npy", allow_pickle=True)
                print(f"DEBUG AF2ClusterBatchSampler: Loaded cluster_sizes with allow_pickle, shape: {self.cluster_sizes.shape}")
            except Exception as e:
                raise RuntimeError(f"Failed to load cluster_sizes from {cluster_sizes_path} or fallback: {e}")

        print(f"DEBUG AF2ClusterBatchSampler: Loading cluster_offsets...")
        try:
            self.cluster_offsets = np.load(cluster_offsets_path, mmap_mode='r')
            print(f"DEBUG AF2ClusterBatchSampler: Loaded cluster_offsets, shape: {self.cluster_offsets.shape}")
        except:
            try:
                self.cluster_offsets = np.load("/workspace/datasets/af_clusters/cluster_offsets.npy", allow_pickle=True)
                print(f"DEBUG AF2ClusterBatchSampler: Loaded cluster_offsets with allow_pickle, shape: {self.cluster_offsets.shape}")
            except Exception as e:
                raise RuntimeError(f"Failed to load cluster_offsets from {cluster_offsets_path} or fallback: {e}")

        print(f"DEBUG AF2ClusterBatchSampler: Loading flat_members...")
        try:
            self.flat_members = np.load(flat_members_path, mmap_mode='r')
            print(f"DEBUG AF2ClusterBatchSampler: Loaded flat_members, shape: {self.flat_members.shape}")
        except:
            try:
                self.flat_members = np.load("/workspace/datasets/af_clusters/flat_members.npy", allow_pickle=True)
                print(f"DEBUG AF2ClusterBatchSampler: Loaded flat_members with allow_pickle, shape: {self.flat_members.shape}")
            except Exception as e:
                raise RuntimeError(f"Failed to load flat_members from {flat_members_path} or fallback: {e}")

        self.num_clusters = len(self.cluster_sizes)
        print(f"DEBUG AF2ClusterBatchSampler: Number of clusters: {self.num_clusters}")
        
        print(f"DEBUG AF2ClusterBatchSampler: Filtering empty clusters...")
        # Filter out empty clusters
        self.valid_cluster_indices = np.where(self.cluster_sizes > 0)[0]
        print(f"DEBUG AF2ClusterBatchSampler: Found {len(self.valid_cluster_indices)} non-empty clusters")
        
        if len(self.valid_cluster_indices) == 0:
            raise ValueError("No valid clusters found (all clusters are empty)")
        
        print(f"Loaded cluster metadata: {self.num_clusters} total clusters, "
              f"{len(self.valid_cluster_indices)} non-empty clusters")
    
    def set_epoch(self, epoch: int):
        """Set the epoch for distributed training synchronization."""
        self.epoch = epoch
    
    def _get_worker_seed(self, batch_idx: int) -> int:
        """Generate worker-specific seed to prevent identical sampling across processes."""
        # Unique seed: base_seed + rank * 10000 + worker_id * 100 + epoch * 1000000 + batch_idx
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        
        return (self.seed + 
                self.rank * 10000 + 
                worker_id * 100 + 
                self.epoch * 1000000 + 
                batch_idx)
    
    def _sample_clusters(self, batch_idx: int) -> np.ndarray:
        """Sample batch_size unique clusters for this batch."""
        # Set worker-specific seed
        worker_seed = self._get_worker_seed(batch_idx)
        rng = np.random.RandomState(worker_seed)
        
        # Sample unique clusters without replacement
        if len(self.valid_cluster_indices) < self.batch_size:
            raise ValueError(f"Not enough valid clusters ({len(self.valid_cluster_indices)}) "
                           f"for batch size ({self.batch_size})")
        
        selected_clusters = rng.choice(self.valid_cluster_indices, 
                                     size=self.batch_size, 
                                     replace=False)
        return selected_clusters
    
    def _sample_member_from_cluster(self, cluster_id: int, rng: np.random.RandomState) -> str:
        """Sample one member UniProt ID from the specified cluster."""
        cluster_size = self.cluster_sizes[cluster_id]
        cluster_offset = self.cluster_offsets[cluster_id]
        
        if cluster_size == 0:
            raise ValueError(f"Cluster {cluster_id} is empty")
        
        # Randomly select one member from this cluster
        member_idx = rng.randint(0, cluster_size)
        global_member_idx = cluster_offset + member_idx
        
        # Ensure we don't exceed array bounds
        if global_member_idx >= len(self.flat_members):
            raise IndexError(f"Global member index {global_member_idx} exceeds flat_members length {len(self.flat_members)}")
        
        member_id = self.flat_members[global_member_idx]
        return str(member_id)  # Ensure string format
    
    def __iter__(self) -> Iterator[List[str]]:
        """
        Generate batches of UniProt IDs.
        
        Yields:
            List[str]: Batch of UniProt IDs, one from each sampled cluster
        """
        print(f"DEBUG AF2ClusterBatchSampler: Starting __iter__ for {self.num_batches_per_epoch} batches")
        
        for batch_idx in range(self.num_batches_per_epoch):
            print(f"DEBUG AF2ClusterBatchSampler: Starting batch {batch_idx+1}/{self.num_batches_per_epoch}")
            
            if self.verbose:
                print(f"AF2ClusterBatchSampler: Starting batch {batch_idx+1}/{self.num_batches_per_epoch}", flush=True)
            
            # Sample unique clusters for this batch
            print(f"DEBUG AF2ClusterBatchSampler: Sampling clusters for batch {batch_idx}")
            selected_clusters = self._sample_clusters(batch_idx)
            print(f"DEBUG AF2ClusterBatchSampler: Selected {len(selected_clusters)} clusters")
            
            if self.verbose:
                print(f"AF2ClusterBatchSampler: Selected clusters {selected_clusters[:3].tolist()}{'...' if len(selected_clusters) > 3 else ''}", flush=True)
            
            # Use same seed for member sampling to ensure reproducibility
            worker_seed = self._get_worker_seed(batch_idx)
            rng = np.random.RandomState(worker_seed + 1)  # +1 to differentiate from cluster sampling
            
            # Sample one member from each selected cluster
            batch_member_ids = []
            print(f"DEBUG AF2ClusterBatchSampler: Sampling members from {len(selected_clusters)} clusters")
            for i, cluster_id in enumerate(selected_clusters):
                print(f"DEBUG AF2ClusterBatchSampler: Sampling from cluster {cluster_id} ({i+1}/{len(selected_clusters)})")
                try:
                    member_id = self._sample_member_from_cluster(cluster_id, rng)
                    batch_member_ids.append(member_id)
                    print(f"DEBUG AF2ClusterBatchSampler: Got member {member_id} from cluster {cluster_id}")
                except (ValueError, IndexError) as e:
                    print(f"ERROR AF2ClusterBatchSampler: Failed to sample from cluster {cluster_id}: {e}")
                    raise RuntimeError(f"Failed to sample from cluster {cluster_id}: {e}")
            
            if self.verbose:
                print(f"AF2ClusterBatchSampler: Sampled UniProt IDs: {batch_member_ids[:3]}{'...' if len(batch_member_ids) > 3 else ''}", flush=True)
            
            yield batch_member_ids
    
    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.num_batches_per_epoch


def create_af2_batch_sampler(cluster_dir: str,
                           batch_size: int,
                           num_batches_per_epoch: int,
                           rank: int = 0,
                           world_size: int = 1,
                           seed: int = 42) -> AF2ClusterBatchSampler:
    """
    Factory function to create AF2ClusterBatchSampler.
    
    Args:
        cluster_dir: Directory containing cluster metadata
        batch_size: Number of clusters per batch
        num_batches_per_epoch: Number of batches per epoch
        rank: Process rank for distributed training
        world_size: Total number of processes
        seed: Base random seed
        
    Returns:
        Configured AF2ClusterBatchSampler
    """
    return AF2ClusterBatchSampler(
        cluster_dir=cluster_dir,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        rank=rank,
        world_size=world_size,
        seed=seed
    )
