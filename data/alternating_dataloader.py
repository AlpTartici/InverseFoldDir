"""
alternating_dataloader.py

DataLoader that creates pure homogeneous batches that alternate between AF2-only 
and PDB-only batches according to a specified ratio.

For example, with ratio_af2_pdb=3:
- 3 pure AF2 batches, then 1 pure PDB batch, then 3 pure AF2 batches, etc.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List, Iterator, Tuple, Any
import random

from .cath_dataset import CathDataset
from .unified_dataset import UnifiedDataset


class AlternatingBatchDataLoader:
    """
    DataLoader that creates alternating pure batches (homogeneous).
    
    Each batch contains only one data type (either all AF2 or all PDB).
    Batches alternate according to the specified ratio.
    """
    
    def __init__(self,
                 # PDB parameters
                 split_json: Optional[str] = None,
                 map_pkl: Optional[str] = None,
                 split: str = 'train',
                 
                 # AF2 parameters
                 af2_chunk_dir: Optional[str] = None,
                 af2_chunk_limit: Optional[int] = None,
                 
                 # Alternating parameters
                 ratio_af2_pdb: int = 3,  # 3 AF2 batches per 1 PDB batch
                 num_batches_per_epoch: Optional[int] = None,  # Limit batches per epoch
                 
                 # DataLoader parameters
                 batch_size: int = 32,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 
                 # Common parameters
                 max_len: Optional[int] = None,
                 graph_builder_kwargs: Optional[Dict] = None,
                 
                 # Iteration control
                 deterministic: bool = True,
                 
                 # Distributed training
                 rank: int = 0,
                 world_size: int = 1,
                 seed: int = 42):
        """
        Initialize alternating batch dataloader.
        
        Args:
            ratio_af2_pdb: Number of AF2 batches per PDB batch
            Other args: Same as unified dataset
        """
        self.ratio_af2_pdb = ratio_af2_pdb
        self.batch_size = batch_size
        self.deterministic = deterministic
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        
        # Create separate datasets for AF2 and PDB
        self.af2_dataset = None
        self.pdb_dataset = None
        
        if af2_chunk_dir:
            # Create AF2-only dataset
            self.af2_dataset = UnifiedDataset(
                split_json=None,
                map_pkl=None,
                split=split,
                af2_chunk_dir=af2_chunk_dir,
                af2_chunk_limit=af2_chunk_limit,
                ratio_af2_pdb=-1,  # AF2-only
                max_len=max_len,
                graph_builder_kwargs=graph_builder_kwargs,
                deterministic=deterministic,
                rank=rank,
                world_size=world_size,
                seed=seed
            )
        
        if split_json and map_pkl:
            # Create PDB-only dataset
            self.pdb_dataset = UnifiedDataset(
                split_json=split_json,
                map_pkl=map_pkl,
                split=split,
                af2_chunk_dir=None,
                ratio_af2_pdb=0,  # PDB-only
                max_len=max_len,
                graph_builder_kwargs=graph_builder_kwargs,
                deterministic=deterministic,
                rank=rank,
                world_size=world_size,
                seed=seed
            )
        
        # Create individual dataloaders
        from training.collate import collate_fn
        
        self.af2_loader = None
        self.pdb_loader = None
        
        if self.af2_dataset:
            self.af2_loader = DataLoader(
                self.af2_dataset,
                batch_size=batch_size,
                shuffle=not deterministic,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn
            )
        
        if self.pdb_dataset:
            self.pdb_loader = DataLoader(
                self.pdb_dataset,
                batch_size=batch_size,
                shuffle=not deterministic,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn
            )
        
        # Calculate effective dataset size for alternating pattern
        af2_batches = len(self.af2_loader) if self.af2_loader else 0
        pdb_batches = len(self.pdb_loader) if self.pdb_loader else 0
        
        if af2_batches > 0 and pdb_batches > 0:
            # Mixed mode: calculate how many complete cycles we can do
            cycle_length = ratio_af2_pdb + 1  # AF2 batches + 1 PDB batch
            max_cycles = min(af2_batches // ratio_af2_pdb, pdb_batches)
            calculated_batches = max_cycles * cycle_length
        elif af2_batches > 0:
            calculated_batches = af2_batches
        elif pdb_batches > 0:
            calculated_batches = pdb_batches
        else:
            calculated_batches = 0
        
        # Apply num_batches_per_epoch limit if specified
        if num_batches_per_epoch is not None:
            self.total_batches = min(calculated_batches, num_batches_per_epoch)
            print(f"Limiting alternating dataloader to {self.total_batches} batches per epoch (calculated: {calculated_batches})")
        else:
            self.total_batches = calculated_batches
        
        print(f"Alternating dataloader: {af2_batches} AF2 batches, {pdb_batches} PDB batches")
        print(f"Alternating pattern: {ratio_af2_pdb} AF2 batches → 1 PDB batch (total: {self.total_batches} batches)")
    
    def __len__(self) -> int:
        """Return total number of alternating batches."""
        return self.total_batches
    
    def __iter__(self) -> Iterator[Tuple[Any, Any, Any]]:
        """Iterate through alternating pure batches."""
        if not self.af2_loader and not self.pdb_loader:
            return
        
        # Create iterators
        af2_iter = iter(self.af2_loader) if self.af2_loader else None
        pdb_iter = iter(self.pdb_loader) if self.pdb_loader else None
        
        batch_count = 0
        af2_in_cycle = 0  # Track AF2 batches in current cycle
        
        while batch_count < self.total_batches:
            try:
                if af2_in_cycle < self.ratio_af2_pdb and af2_iter:
                    # Yield AF2 batch
                    batch = next(af2_iter)
                    af2_in_cycle += 1
                    batch_type = "AF2"
                    
                elif pdb_iter:
                    # Yield PDB batch and reset cycle
                    batch = next(pdb_iter)
                    af2_in_cycle = 0  # Reset cycle
                    batch_type = "PDB"
                    
                else:
                    # No more batches available
                    break
                
                # Add batch type info for debugging
                data, y, mask = batch
                if hasattr(data, 'batch_info'):
                    data.batch_info['batch_type'] = batch_type
                else:
                    data.batch_info = {'batch_type': batch_type}
                
                yield batch
                batch_count += 1
                
            except StopIteration:
                # One of the iterators is exhausted
                break
    
    def reset_epoch(self):
        """Reset for new epoch (if datasets support it)."""
        if hasattr(self.af2_dataset, 'reset_iteration'):
            self.af2_dataset.reset_iteration()
        if hasattr(self.pdb_dataset, 'reset_iteration'):
            self.pdb_dataset.reset_iteration()


def create_alternating_dataloader(
    # PDB parameters
    split_json: Optional[str] = None,
    map_pkl: Optional[str] = None,
    split: str = 'train',
    
    # AF2 parameters
    af2_chunk_dir: Optional[str] = None,
    af2_chunk_limit: Optional[int] = None,
    
    # Alternating parameters
    ratio_af2_pdb: int = 3,
    num_batches_per_epoch: Optional[int] = None,  # Limit batches per epoch
    
    # DataLoader parameters
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    
    # Common parameters
    max_len: Optional[int] = None,
    graph_builder_kwargs: Optional[Dict] = None,
    
    # Iteration control
    deterministic: bool = True,
    
    # Distributed training
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42) -> AlternatingBatchDataLoader:
    """
    Create alternating batch dataloader for pure homogeneous batches.
    
    Returns:
        AlternatingBatchDataLoader that yields pure AF2 or PDB batches in alternating pattern
    """
    return AlternatingBatchDataLoader(
        split_json=split_json,
        map_pkl=map_pkl,
        split=split,
        af2_chunk_dir=af2_chunk_dir,
        af2_chunk_limit=af2_chunk_limit,
        ratio_af2_pdb=ratio_af2_pdb,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        max_len=max_len,
        graph_builder_kwargs=graph_builder_kwargs,
        deterministic=deterministic,
        rank=rank,
        world_size=world_size,
        seed=seed
    )
