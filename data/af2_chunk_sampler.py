"""
af2_chunk_sampler.py

AlphaFold2 chunk-based batch sampler for scalable protein training.
Sequential iteration through all chunks, extracting multiple batches per chunk.
"""
import os
from typing import Iterator, Tuple


class AF2ChunkBatchSampler:
    """
    Sequential batch sampler for AlphaFold2 chunk-based iteration.

    Iterates through chunks sequentially:
    1. group_1/af2_chunk_000000.pkl → group_2/af2_chunk_000000.pkl → ... → group_8/af2_chunk_000000.pkl
    2. group_1/af2_chunk_000001.pkl → group_2/af2_chunk_000001.pkl → ... → group_8/af2_chunk_000001.pkl
    3. Continue until group_8/af2_chunk_000689.pkl

    For each chunk file:
    - Load once and extract multiple full batches from it
    - Discard leftover proteins that don't make a full batch
    """

    def __init__(self,
                 chunk_dir: str,
                 batch_size: int,
                 rank: int = 0,
                 world_size: int = 1):
        """
        Initialize AF2 sequential chunk batch sampler.

        Args:
            chunk_dir: Directory containing group_X subdirectories with pickle chunks
            batch_size: Number of proteins per batch
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.chunk_dir = chunk_dir
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        # Sequential iteration state
        self.current_group = 1
        self.current_chunk = 0
        self.current_chunk_proteins = None
        self.current_chunk_path = None
        self.proteins_used_from_current_chunk = 0

        # Constants for AF2 structure
        self.max_group = 8
        self.max_chunk = 689

        # Validate chunk directory exists
        if not os.path.exists(chunk_dir):
            raise FileNotFoundError(f"AF2 chunk directory does not exist: {chunk_dir}")

        print(f"AF2ChunkBatchSampler (sequential) initialized: chunk_dir={chunk_dir}, "
              f"batch_size={batch_size}, rank={rank}/{world_size}")

    def set_epoch(self, epoch: int):
        """Set the epoch and reset iteration state."""
        self.epoch = epoch
        self.current_group = 1
        self.current_chunk = 0
        self.current_chunk_proteins = None
        self.current_chunk_path = None
        self.proteins_used_from_current_chunk = 0
        print(f"Starting epoch {epoch} - resetting to group_1/af2_chunk_000000.pkl")

    def _get_current_chunk_path(self) -> str:
        """Get the current chunk file path."""
        chunk_filename = f"af2_chunk_{self.current_chunk:06d}.pkl"
        chunk_path = os.path.join(self.chunk_dir, f"group_{self.current_group}", chunk_filename)
        return chunk_path

    def _advance_to_next_chunk(self):
        """Move to the next chunk in sequential order."""
        self.current_group += 1
        if self.current_group > self.max_group:
            self.current_group = 1
            self.current_chunk += 1
            if self.current_chunk > self.max_chunk:
                # End of epoch - start over
                self.current_chunk = 0

        # Reset chunk state
        self.current_chunk_proteins = None
        self.current_chunk_path = None
        self.proteins_used_from_current_chunk = 0

    def __iter__(self) -> Iterator[Tuple[str, int, int]]:
        """
        Iterate over batches sequentially through all chunks.
        Yields (chunk_path, start_idx, batch_size) for each batch.
        """
        batches_yielded = 0

        while True:
            chunk_path = self._get_current_chunk_path()

            # For the very first batch, we start with current position
            if batches_yielded == 0:
                print(f"Starting with {chunk_path}")

            # Check if we need to advance to next chunk (when we haven't "loaded" this chunk yet)
            if self.current_chunk_proteins is None:
                # This indicates we need to signal that a new chunk should be loaded
                self.current_chunk_path = chunk_path
                self.proteins_used_from_current_chunk = 0

                # Mark chunk as "ready to load" - actual loading happens in dataloader
                self.current_chunk_proteins = "loading"  # Placeholder to indicate loading state

            # Yield batch info: (chunk_path, start_index, batch_size)
            # The actual chunk size validation happens in the dataset/dataloader
            start_idx = self.proteins_used_from_current_chunk
            yield (chunk_path, start_idx, self.batch_size)

            # Update state for next batch
            self.proteins_used_from_current_chunk += self.batch_size
            batches_yielded += 1

    def advance_to_next_chunk(self):
        """
        Signal that current chunk is exhausted and we should move to next.
        This is called by the dataloader when it detects insufficient proteins.
        """
        self._advance_to_next_chunk()

    def __len__(self) -> int:
        """
        Return estimated number of batches per epoch.
        This is an approximation since we don't know protein counts per chunk.
        """
        total_chunks = self.max_group * (self.max_chunk + 1)  # 8 * 690 = 5520 chunks
        estimated_proteins_per_chunk = 1500  # Conservative estimate
        estimated_batches_per_chunk = estimated_proteins_per_chunk // self.batch_size
        return total_chunks * estimated_batches_per_chunk
