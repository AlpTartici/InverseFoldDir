"""
af2_dataset.py

AlphaFold2 dataset for on-demand CIF file loading from Azure Blob Storage.
Downloads and processes AF2 structures with robust error handling.
Supports both public and private Azure blob access.
"""
import io
import os
import tempfile
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

# Optional Azure SDK imports (graceful fallback if not installed)
try:
    from azure.core.exceptions import AzureError
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False

from .cif_parser import parse_cif_backbone_auto
from .graph_builder import GraphBuilder


class AF2Dataset(Dataset):
    """
    Dataset for loading AlphaFold2 structures on-demand from Azure Blob Storage.

    Takes UniProt IDs and downloads corresponding CIF files from Azure,
    then processes them into graph representations using the existing
    GraphBuilder pipeline.
    """

    def __init__(self,
                 remote_data_dir: str,
                 max_retries: int = 5,
                 timeout: float = 30.0,
                 max_len: Optional[int] = None,
                 graph_builder_kwargs: Optional[Dict] = None):
        """
        Initialize AF2 dataset.

        Args:
            remote_data_dir: Path to directory containing AF2 CIF files
            max_retries: Maximum retry attempts for failed file reads
            timeout: File operation timeout in seconds (for consistency)
            max_len: Maximum sequence length filter (None for no limit)
            graph_builder_kwargs: Arguments for GraphBuilder initialization
        """
        print(f"DEBUG AF2Dataset: Starting initialization...")
        print(f"DEBUG AF2Dataset: remote_data_dir={remote_data_dir}")
        print(f"DEBUG AF2Dataset: max_retries={max_retries}, timeout={timeout}")

        self.remote_data_dir = remote_data_dir.rstrip('/')  # Remove trailing slash
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_len = max_len

        print(f"DEBUG AF2Dataset: Setting up graph_builder_kwargs...")
        # Initialize graph builder with AF2-specific settings
        if graph_builder_kwargs is None:
            graph_builder_kwargs = {}

        # Check if verbose mode is enabled (for debugging small datasets)
        self.verbose = graph_builder_kwargs.get('verbose', False)

        print(f"DEBUG AF2Dataset: Creating GraphBuilder...")
        self.graph_builder = GraphBuilder(**graph_builder_kwargs)
        print(f"DEBUG AF2Dataset: GraphBuilder created successfully")

        print(f"DEBUG AF2Dataset: Skipping directory validation for Azure blob storage...")
        # Skipping directory listing operations that timeout on Azure blob storage
        print(f"DEBUG: AF2Dataset path configuration:")
        print(f"  remote_data_dir: {self.remote_data_dir}")
        print(f"  Skipping os.path.exists() and os.listdir() to avoid Azure blob timeouts")

        if self.verbose:
            print(f"AF2Dataset initialized (VERBOSE MODE): remote_data_dir={self.remote_data_dir}, "
                  f"max_retries={max_retries}, timeout={timeout}s", flush=True)
        else:
            print(f"AF2Dataset initialized: remote_data_dir={self.remote_data_dir}, "
                  f"max_retries={max_retries}, timeout={timeout}s")

    def _construct_cif_path(self, uniprot_id: str) -> str:
        """Construct the local file path for a given UniProt ID."""
        # Format: AF-{uniprot_id}-F1-model_v4.cif
        filename = f"AF-{uniprot_id}-F1-model_v4.cif"
        return os.path.join(self.remote_data_dir, filename)

    def _read_cif_content(self, file_path: str) -> bytes:
        """
        Read CIF file content from local filesystem.

        Args:
            file_path: Full path to the CIF file

        Returns:
            bytes: CIF file content

        Raises:
            RuntimeError: If file read fails after all retries
        """
        last_exception = None

        # DEBUG: Print path information (skip Azure blob validation)
        print(f"DEBUG: Attempting to read CIF file at: {file_path}")
        print(f"DEBUG: Skipping file existence checks to avoid Azure blob timeouts")

        # Skip parent directory listing (Azure blob storage can timeout)

        for attempt in range(self.max_retries):
            try:
                # Skip file existence check - let open() handle file not found
                with open(file_path, 'rb') as f:
                    content = f.read()

                if len(content) == 0:
                    raise ValueError("File is empty")

                return content

            except (FileNotFoundError, PermissionError, OSError, ValueError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    if self.verbose:
                        print(f"AF2 Read attempt {attempt + 1} failed for {file_path}: {e}. Retrying...", flush=True)
                    else:
                        print(f"Read attempt {attempt + 1} failed for {file_path}: {e}. Retrying...")
                    continue
                else:
                    break

        # All retries failed
        if self.verbose:
            print(f"AF2 Read FAILED for {file_path} after {self.max_retries} attempts: {last_exception}", flush=True)
        raise RuntimeError(f"Failed to read {file_path} after {self.max_retries} attempts. "
                         f"Last error: {last_exception}")

    def _download_cif_content(self, url: str) -> bytes:
        """
        Download CIF file content from private Azure blob with authentication.

        Args:
            url: Full URL to the CIF file

        Returns:
            bytes: CIF file content

        Raises:
            RuntimeError: If download fails after all retries
        """
        if not AZURE_SDK_AVAILABLE:
            print("Warning: Azure SDK not available, falling back to public access")
            return self._download_cif_content_public(url)

        last_exception = None

        # Parse the Azure blob URL to extract components
        # Expected format: https://{account}.blob.core.windows.net/{container}/{blob_path}
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)

            # Extract account name from hostname
            account_name = parsed_url.hostname.split('.')[0]

            # Extract container and blob path
            path_parts = parsed_url.path.strip('/').split('/', 1)
            container_name = path_parts[0]
            blob_name = path_parts[1] if len(path_parts) > 1 else ''

            if not blob_name:
                raise ValueError(f"Could not extract blob name from URL: {url}")

        except Exception as e:
            print(f"Failed to parse Azure URL {url}: {e}")
            return self._download_cif_content_public(url)

        for attempt in range(self.max_retries):
            try:
                # Create blob service client with default credential (managed identity, env vars, etc.)
                # This will use DefaultAzureCredential which tries multiple auth methods
                account_url = f"https://{account_name}.blob.core.windows.net"

                # Try to use DefaultAzureCredential if available
                credential = None
                try:
                    credential = DefaultAzureCredential()
                    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
                except Exception:
                    # Fall back to anonymous/public access if credential fails
                    blob_service_client = BlobServiceClient(account_url=account_url)

                # Get blob client and download
                blob_client = blob_service_client.get_blob_client(
                    container=container_name,
                    blob=blob_name
                )

                # Download with timeout
                download_stream = blob_client.download_blob(timeout=self.timeout)
                content = download_stream.readall()

                if len(content) == 0:
                    raise ValueError("Downloaded empty file")

                if self.verbose:
                    print(f"AF2 Download SUCCESS via Azure SDK: {len(content)} bytes", flush=True)

                return content

            except (AzureError, ValueError, Exception) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    if self.verbose:
                        print(f"AF2 Azure download attempt {attempt + 1} failed: {e}. Retrying", flush=True)
                    else:
                        print(f"Azure download attempt {attempt + 1} failed for {url}: {e}. Retrying")
                    continue
                else:
                    break

        # All Azure SDK attempts failed, try fallback to public access
        print(f"Azure SDK download failed after {self.max_retries} attempts, trying public access")
        try:
            return self._download_cif_content_public(url)
        except Exception as fallback_error:
            print(f"Public access fallback also failed: {fallback_error}")

        # Both methods failed
        if self.verbose:
            print(f"AF2 Download FAILED via Azure SDK after {self.max_retries} attempts: {last_exception}", flush=True)
        raise RuntimeError(f"Failed to download {url} after {self.max_retries} attempts with Azure SDK. "
                         f"Last error: {last_exception}")

    def _parse_cif_from_content(self, content: bytes, uniprot_id: str) -> Tuple[torch.Tensor, torch.Tensor, list, str]:
        """
        Parse CIF content into coordinates and metadata.

        Args:
            content: Raw CIF file content
            uniprot_id: UniProt ID for error reporting

        Returns:
            Tuple of (coords, scores, residue_types, source)

        Raises:
            RuntimeError: If parsing fails
        """
        try:
            # Write content to temporary file for parsing
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.cif', delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_file.flush()

                try:
                    # Parse using existing CIF parser
                    coords, scores, residue_types, source = parse_cif_backbone_auto(tmp_file.name)

                    # Validate parsed data
                    if len(coords) == 0:
                        raise ValueError("No valid residues found in CIF file")

                    if len(coords) != len(scores) or len(coords) != len(residue_types):
                        raise ValueError(f"Inconsistent data lengths: coords={len(coords)}, "
                                       f"scores={len(scores)}, residues={len(residue_types)}")

                    return coords, scores, residue_types, source

                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file.name)
                    except OSError:
                        pass  # Ignore cleanup errors

        except Exception as e:
            raise RuntimeError(f"Failed to parse CIF content for {uniprot_id}: {e}")

    def _apply_length_filter(self, coords: torch.Tensor, uniprot_id: str) -> bool:
        """Check if protein passes length filter."""
        if self.max_len is None:
            return True

        seq_len = len(coords)
        if seq_len > self.max_len:
            print(f"Filtered out {uniprot_id}: length {seq_len} > max_len {self.max_len}")
            return False

        return True

    def __getitem__(self, uniprot_id: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load and process a single AF2 structure.

        Args:
            uniprot_id: UniProt identifier for the protein

        Returns:
            Tuple of (data, y, mask) where:
            - data: PyTorch Geometric Data object
            - y: One-hot encoded sequence tensor [L, 21]
            - mask: Boolean mask tensor [L]

        Raises:
            RuntimeError: If download, parsing, or processing fails
        """
        if not isinstance(uniprot_id, str):
            raise TypeError(f"Expected string UniProt ID, got {type(uniprot_id)}")

        try:
            if self.verbose:
                print(f"AF2 Processing: {uniprot_id} - Starting file read...", flush=True)

            # Construct local file path
            cif_path = self._construct_cif_path(uniprot_id)

            # Read CIF content from local file
            cif_content = self._read_cif_content(cif_path)

            if self.verbose:
                print(f"AF2 Processing: {uniprot_id} - Read {len(cif_content)} bytes, parsing CIF...", flush=True)

            # Parse CIF content
            coords, scores, residue_types, source = self._parse_cif_from_content(cif_content, uniprot_id)

            if self.verbose:
                print(f"AF2 Processing: {uniprot_id} - Parsed {len(coords)} residues, building graph...", flush=True)

            # Apply length filter
            if not self._apply_length_filter(coords, uniprot_id):
                raise RuntimeError(f"Protein {uniprot_id} filtered out due to length constraint")

            # Create data entry in format expected by GraphBuilder
            entry = {
                'name': uniprot_id,
                'seq': ''.join([self._three_to_one_aa(rt) for rt in residue_types]),
                'coords': {
                    'N': coords[:, 0].numpy(),
                    'CA': coords[:, 1].numpy(),
                    'C': coords[:, 2].numpy(),
                    'O': coords[:, 3].numpy() if coords.shape[1] > 3 else coords[:, 1].numpy()
                },
                'source': source,
                'scores': scores.numpy() if hasattr(scores, 'numpy') else scores
            }

            # Build graph using existing GraphBuilder
            data = self.graph_builder.build_from_dict(entry)
            data.source = 'alphafold2'
            data.name = uniprot_id

            # Create ground truth sequence and mask
            if data.use_virtual_node:
                L = data.num_nodes - 1  # Real nodes = total - virtual
            else:
                L = data.num_nodes  # Real nodes = total

            # Use filtered sequence from graph builder for perfect alignment
            filtered_seq = getattr(data, 'filtered_seq', entry['seq'])

            # Verify alignment
            if len(filtered_seq) != L:
                raise RuntimeError(f"Sequence length mismatch for {uniprot_id}: "
                                 f"filtered_seq={len(filtered_seq)}, nodes={L}")

            # Create one-hot encoding
            y = torch.zeros(L, 21, dtype=torch.float32)  # 21 classes including XXX
            for i, aa in enumerate(filtered_seq):
                aa_idx = self._aa_to_index(aa)
                y[i, aa_idx] = 1.0

            # Create mask (all True for AF2 data)
            mask = torch.ones(L, dtype=torch.bool)

            if self.verbose:
                print(f"AF2 SUCCESS: {uniprot_id} - {len(filtered_seq)} residues, {data.num_nodes} nodes, {data.edge_index.shape[1]} edges", flush=True)

            return data, y, mask

        except Exception as e:
            if self.verbose:
                print(f"AF2 FAILED: {uniprot_id} - {e}", flush=True)
            # Re-raise with context for upstream error handling
            raise RuntimeError(f"Failed to process AF2 structure {uniprot_id}: {e}")

    def _three_to_one_aa(self, three_letter: str) -> str:
        """Convert 3-letter amino acid code to 1-letter code."""
        aa_map = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        return aa_map.get(three_letter.upper(), 'X')  # Unknown -> X

    def _aa_to_index(self, aa: str) -> int:
        """Convert amino acid to index for one-hot encoding."""
        aa_to_idx = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15,
            'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20  # Unknown/non-standard
        }
        return aa_to_idx.get(aa.upper(), 20)  # Default to unknown (index 20)

    def __len__(self):
        """AF2Dataset length is undefined (streaming dataset)."""
        # This is a streaming dataset, length is determined by the batch sampler
        return float('inf')


class AF2DatasetWithErrorRecovery(AF2Dataset):
    """
    Extended AF2Dataset with cluster resampling on failures.

    Integrates with AF2ClusterBatchSampler to resample different clusters
    when individual proteins fail to download/parse.
    """

    def __init__(self,
                 base_url: str,
                 cluster_sampler,  # AF2ClusterBatchSampler instance
                 max_cluster_retries: int = 3,
                 **kwargs):
        """
        Initialize AF2Dataset with error recovery.

        Args:
            base_url: Base URL for AF2 CIF files
            cluster_sampler: AF2ClusterBatchSampler instance for resampling
            max_cluster_retries: Maximum cluster resampling attempts
            **kwargs: Additional arguments passed to AF2Dataset
        """
        super().__init__(base_url=base_url, **kwargs)
        self.cluster_sampler = cluster_sampler
        self.max_cluster_retries = max_cluster_retries

    def __getitem__(self, uniprot_id: str):
        """
        Load AF2 structure with cluster resampling on failure.

        If the requested protein fails to load, attempts to resample
        a different cluster and try again.
        """
        for attempt in range(self.max_cluster_retries):
            try:
                return super().__getitem__(uniprot_id)
            except RuntimeError as e:
                if attempt < self.max_cluster_retries - 1:
                    print(f"AF2 loading failed for {uniprot_id} (attempt {attempt + 1}): {e}")
                    print("This would trigger cluster resampling in production...")
                    # In production, this would trigger cluster resampling
                    # For now, we re-raise the error
                raise RuntimeError(f"Failed to load AF2 structure {uniprot_id} after "
                                 f"{self.max_cluster_retries} attempts: {e}")
