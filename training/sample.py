#!/usr/bin/env python3
"""
Core protein sequence sampling using trained DFM models.

Contains the essential sampling algorithms and main entry point.
Utility functions are in sample_utils.py.
"""

import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from tqdm import tqdm

# Add the parent directory to the Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import utilities from the separate utils module
from training.sample_utils import (
    AA_TO_IDX,
    IDX_TO_AA,
    SINGLE_TO_TRIPLE,
    THREE_TO_ONE,
    adjust_batch_size_for_ensemble,
    compute_sampling_metrics,
    create_argument_parser,
    create_structural_ensemble,
    generate_detailed_json_output,
    resolve_protein_sampling_mode,
    sample_multiple_proteins,
    sample_multiple_proteins_batched,
    sample_multiple_proteins_with_ensemble,
    sample_multiple_proteins_with_trajectory,
    sample_with_ensemble_consensus,
    save_results_to_files,
    simplex_proj,
)


def sample_chain(model, data, dataset, structure_idx=None, T=8.0, t_min=0.0, steps=20, K=21, verbose=False, args=None):
    """
    Generates an amino acid sequence for a single protein structure using the
    trained DFM model. This is the reverse sampling process.

    This function starts with a random sequence and iteratively denoises it over
    a series of time steps by following the velocity field predicted by the model.

    Args:
        model (DFMNodeClassifier): The trained model.
        data (torch_geometric.data.Data): The graph representation of the protein structure.
        dataset: The dataset object to access original entries for graph rebuilding.
        structure_idx: Index of the structure in the dataset (for graph rebuilding).
        T (float): The starting time (maximum noise level). Should match the training setup.
        t_min (float): The minimum time (initial noise level).
        steps (int): The number of denoising steps to perform.
        K (int): Number of amino acid classes (default: 21).
        verbose (bool): If True, prints detailed velocity and c factor debugging information.
        args: Arguments object with sampling parameters.

    Returns:
        tuple: (final_probabilities, predicted_sequence, evaluation_metrics)
    """
    if verbose:
        print("="*80)
        print("CORE SAMPLING: Single Chain Generation")
        print("="*80)

    # Ensure model is on the correct device and in evaluation mode
    device = next(model.parameters()).device
    model.eval()

    # Extract structure noise parameters from args, defaulting to 0.0 if not provided
    structure_noise_mag_std = getattr(args, 'structure_noise_mag_std', 0.0)
    if structure_noise_mag_std is None:
        structure_noise_mag_std = 0.0

    time_based_struct_noise = getattr(args, 'time_based_struct_noise', 'fixed')
    uncertainty_struct_noise_scaling = getattr(args, 'uncertainty_struct_noise_scaling', False)

    # Create graph builder for graph rebuilding at each step (if noise is enabled)
    graph_builder = None
    original_entry = None

    if structure_noise_mag_std > 0 and dataset is not None and structure_idx is not None:
        # Get the original entry from the dataset for rebuilding graphs
        # Handle different dataset types
        if hasattr(dataset, 'entries'):
            # CathDataset has entries list
            original_entry = dataset.entries[structure_idx]
        elif hasattr(dataset, 'protein_entry'):
            # SingleProteinDataset has single protein_entry
            original_entry = dataset.protein_entry
        else:
            raise AttributeError(f"Unknown dataset type: {type(dataset)}")

        # Extract graph builder parameters from the existing graph builder
        existing_gb = dataset.graph_builder

        # Create a new graph builder with noise parameters
        from data.graph_builder import GraphBuilder
        graph_builder = GraphBuilder(
            k=getattr(existing_gb, 'k', None),
            k_farthest=getattr(existing_gb, 'k_farthest', None),
            k_random=getattr(existing_gb, 'k_random', None),
            max_edge_dist=getattr(existing_gb, 'max_edge_dist', None),
            num_rbf_3d=getattr(existing_gb, 'num_rbf_3d', None),
            num_rbf_seq=getattr(existing_gb, 'num_rbf_seq', None),
            use_virtual_node=getattr(existing_gb, 'use_virtual_node', True),
            no_source_indicator=getattr(existing_gb, 'no_source_indicator', False),
            rbf_3d_min=getattr(existing_gb, 'rbf_3d_min', 2.0),
            rbf_3d_max=getattr(existing_gb, 'rbf_3d_max', 350.0),
            rbf_3d_spacing=getattr(existing_gb, 'rbf_3d_spacing', 'exponential'),
            structure_noise_mag_std=structure_noise_mag_std,
            time_based_struct_noise=time_based_struct_noise,
            uncertainty_struct_noise_scaling=uncertainty_struct_noise_scaling,
            verbose=verbose
        )

        if verbose:
            print(f"Structure noise enabled: std={structure_noise_mag_std}, "
                  f"time_based={time_based_struct_noise}, "
                  f"uncertainty_scaling={uncertainty_struct_noise_scaling}")

    # Handle virtual nodes: the last node in each graph is virtual
    # Get the actual structure length from the original data
    total_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x_s.size(0)

    # Determine whether virtual node is used - check both data attribute and global configuration
    use_virtual_node = getattr(data, 'use_virtual_node', False)

    if use_virtual_node:
        N = total_nodes - 1  # Exclude virtual node
        if verbose:
            print(f"Using virtual nodes: {total_nodes} total nodes, {N} real residues")
    else:
        N = total_nodes
        if verbose:
            print(f"No virtual nodes: {N} residues")

    B = 1  # Single sample

    if N <= 0:
        raise ValueError(f"No real nodes found. Total nodes: {total_nodes}, use_virtual_node: {use_virtual_node}")

    if verbose:
        print(f"Sampling parameters: T={T}, t_min={t_min}, steps={steps}, N={N}, K={K}")

    # Start with Dirichlet noise with parameters all 1 (uniform on simplex)
    # This is the proper way to sample from the simplex
    dirichlet_concentration = args.dirichlet_concentration if args and hasattr(args, 'dirichlet_concentration') else 1.0
    dirichlet_dist = Dirichlet(dirichlet_concentration * torch.ones(K, device=device))
    x = dirichlet_dist.sample((1, N))  # Shape: [1, N, K]

    # Debug: Check initial distribution
    if verbose:
        print(f"Initial Dirichlet distribution:")
        print(f"  Shape: {x.shape}")
        print(f"  Concentration: {dirichlet_concentration}")
        print(f"  Min prob: {x.min().item():.6f}")
        print(f"  Max prob: {x.max().item():.6f}")
        print(f"  Mean prob: {x.mean().item():.6f}")

    # Define the discrete time steps for the reverse process
    # In Dirichlet Flow Matching, higher t = more concentrated (cleaner)
    # So we need to go from low t (more noisy) to high t (cleaner)
    times = torch.linspace(t_min, T, steps, device=device)  # Go from t_min to T (e.g., 1.0 to 8.0)
    dt = (T - t_min) / (steps - 1)  # Positive because we're going from low t to high t

    if verbose:
        print(f"Time integration: t={t_min:.1f} -> {T:.1f}, dt={dt:.4f}")
        if args and getattr(args, 'time_as_temperature', False):
            print(f"Using time-dependent temperature: temp = {T:.1f} - current_time + 0.1")
            print(f"  Initial temperature: {T - t_min + 0.1:.1f}")
            print(f"  Final temperature: {T - T + 0.1:.1f}")

    # Initialize batched_data with the original data
    from training.collate import collate_fn

    # Create a dummy batch with just one structure
    # We need to provide (data, y_dummy, mask_dummy, time_dummy) format expected by collate_fn
    dummy_y = torch.zeros(1, K)  # Dummy ground truth (not used in sampling)
    dummy_mask = torch.ones(1, dtype=torch.bool)  # Dummy mask (not used in sampling)
    dummy_time = torch.tensor(0.0)  # Dummy time value (not used in sampling)

    # Use collate_fn to properly batch the single structure
    batched_data, y_pad, mask_pad, time_batch = collate_fn([(data, dummy_y, dummy_mask, dummy_time)])
    batched_data = batched_data.to(device)

    with torch.no_grad():
        time_steps = tqdm(enumerate(times), total=len(times), desc="Sampling time steps", disable=not verbose)
        for i, t_val in time_steps:
            t = torch.full((1,), t_val, device=device)

            # Add structure noise and rebuild graph if enabled
            if graph_builder is not None and original_entry is not None:
                # Rebuild the graph with coordinate noise for the current time step
                try:
                    noisy_data = graph_builder.build_from_dict(original_entry, time_param=t_val.item())
                    noisy_data = noisy_data.to(device)

                    # Re-batch the noisy data
                    batched_data, _, _, _ = collate_fn([(noisy_data, dummy_y, dummy_mask, dummy_time)])
                    batched_data = batched_data.to(device)

                    if verbose and i == 0:
                        print(f"  Graph rebuilt with noise at t={t_val:.3f}")

                except Exception as e:
                    if verbose:
                        print(f"  Warning: Failed to rebuild graph with noise at t={t_val:.3f}: {e}")
                        print("  Continuing with original graph...")

            if verbose and (i + 1) % 10 == 0:
                current_pred = x.argmax(-1).squeeze(0).cpu().tolist()
                current_entropy = -(x * torch.log(x + 1e-10)).sum(-1).mean().item()
                current_max_prob = x.max(-1)[0].mean().item()
                # Show current temperature if using time-dependent temperature
                if args and getattr(args, 'time_as_temperature', False):
                    current_temp = T - t_val + 0.1
                    time_steps.set_postfix({
                        't': f'{t_val:.3f}',
                        'temp': f'{current_temp:.3f}',
                        'entropy': f'{current_entropy:.4f}',
                        'avg_max_prob': f'{current_max_prob:.4f}'
                    })
                else:
                    time_steps.set_postfix({
                        't': f'{t_val:.3f}',
                        'entropy': f'{current_entropy:.4f}',
                        'avg_max_prob': f'{current_max_prob:.4f}'
                    })
            else:
                # Update progress bar with basic info
                time_steps.set_postfix({'t': f'{t_val:.3f}'})

            # Skip velocity prediction on last step
            if i == len(times) - 1:
                break

            # UPDATED FOR POSITION PREDICTION: Get predicted target from model
            model_output = model(batched_data, t, x)

            # Handle DSSP multitask output - model might return (sequence_logits, dssp_logits) tuple
            if isinstance(model_output, tuple):
                position_logits = model_output[0]  # Use only sequence logits for sampling
                if verbose and i == 0:  # Only print once to avoid spam
                    print(f"  Model returns tuple (sequence + DSSP), using sequence logits for sampling")
            else:
                position_logits = model_output

            # DEBUG: Add detailed prediction analysis
            if verbose and (i + 1) % 5 == 0:
                print(f"  [DEBUG STEP {i+1}] t={t_val:.3f} Raw logits analysis:")
                print(f"    Shape: {position_logits.shape}")
                print(f"    Min: {position_logits.min().item():.4f}")
                print(f"    Max: {position_logits.max().item():.4f}")
                print(f"    Mean: {position_logits.mean().item():.4f}")
                print(f"    Std: {position_logits.std().item():.4f}")




            # Convert to predicted target distribution
            # Apply time-dependent temperature if requested
            if args and getattr(args, 'time_as_temperature', False):
                # Temperature starts high (at t_min) and decreases as we approach t_max
                flow_temp = T - t_val + 0.1
            else:
                flow_temp = args.flow_temp if args else 1.0

            predicted_target = torch.softmax(position_logits/flow_temp, dim=-1)

            # Extract only real node predictions and ensure same shape as x
            if use_virtual_node:
                # position_logits has shape [total_nodes, K], need to slice out real nodes only
                # For single structure: real nodes are indices 0 to N-1, virtual node is at index N
                predicted_target_real = predicted_target[:N, :].unsqueeze(0)  # [N, K] -> [1, N, K]
            else:
                # If no virtual node, still need to add batch dimension to match x shape
                predicted_target_real = predicted_target.unsqueeze(0)  # [N, K] -> [1, N, K]

            # Compute analytical velocity using conditional flow
            cond_flow = model.cond_flow
            v_analytical = cond_flow.velocity(
                x,
                predicted_target_real,
                t,
                use_virtual_node=use_virtual_node,
                use_smoothed_targets=getattr(args, 'use_smoothed_targets', False),
                use_c_factor=getattr(args, 'use_c_factor', False)
            )

            # Use analytical velocity for integration
            v_processed = v_analytical

            # Update sequence using Euler step with proper simplex projection
            x_new = x + dt * v_processed
            x_new = simplex_proj(x_new)  # Use proper simplex projection
            x = x_new

    # Return both raw probabilities and argmax predictions
    final_probabilities = x.squeeze(0)  # Shape: [N, K] - probabilities for each residue
    predicted_sequence = final_probabilities.argmax(-1).tolist()  # Argmax predictions

    # Compute evaluation metrics if ground truth is available
    evaluation_metrics = {}
    try:
        # Get ground truth sequence from the same source used for accuracy in other functions
        ground_truth_onehot = None

        # Try multiple sources for ground truth sequence
        if hasattr(data, 'filtered_seq') and data.filtered_seq is not None:
            # Use filtered sequence from graph builder (preferred)
            filtered_seq = data.filtered_seq[:N]  # Exclude virtual node if present

            # Convert to one-hot tensor
            ground_truth_onehot = torch.zeros(N, K, device=device)
            for i, aa in enumerate(filtered_seq):
                if aa in SINGLE_TO_TRIPLE:
                    aa3 = SINGLE_TO_TRIPLE[aa]
                    if aa3 in AA_TO_IDX:
                        ground_truth_onehot[i, AA_TO_IDX[aa3]] = 1.0
                    else:
                        ground_truth_onehot[i, 20] = 1.0  # Unknown
                else:
                    ground_truth_onehot[i, 20] = 1.0  # Unknown
        elif hasattr(data, 'y') and data.y is not None:
            # Use y attribute (already one-hot)
            ground_truth_onehot = data.y[:N].to(device)
        elif hasattr(data, 'aa_seq') and data.aa_seq is not None:
            # Convert amino acid sequence to one-hot
            aa_seq = data.aa_seq[:N]
            ground_truth_onehot = torch.zeros(N, K, device=device)
            for i, aa_idx in enumerate(aa_seq):
                if 0 <= aa_idx < K:
                    ground_truth_onehot[i, aa_idx] = 1.0

        # Compute metrics if ground truth was found
        if ground_truth_onehot is not None and ground_truth_onehot.sum().item() > 0:
            evaluation_metrics = compute_sampling_metrics(
                final_probabilities, ground_truth_onehot, data, model, args, device, use_virtual_node, K
            )
    except Exception as e:
        if verbose:
            print(f"Warning: Could not compute evaluation metrics: {e}")
        evaluation_metrics = {'accuracy': 0.0, 'cce_loss_hard': float('inf'), 'cce_loss_smooth': float('inf')}

    if verbose:
        print(f"\nFINAL RESULTS:")
        print(f"  Final shape: {final_probabilities.shape}")
        print(f"  Final entropy: {-(final_probabilities * torch.log(final_probabilities + 1e-10)).sum(-1).mean().item():.6f}")
        print(f"  Final avg max prob: {final_probabilities.max(-1)[0].mean().item():.6f}")
        print(f"  Final sequence: {predicted_sequence}")
        print(f"  Sequence length: {len(predicted_sequence)}")

    return final_probabilities, predicted_sequence, evaluation_metrics





def sample_chain_with_trajectory(model, data, T=8.0, t_min=0.0, steps=20, K=21, verbose=False, args=None):
    """
    Generates an amino acid sequence for a single protein structure using the
    trained DFM model while tracking the full trajectory.

    This version captures detailed information at each time step including
    the most likely amino acid and its probability for each position.

    Args:
        model (DFMNodeClassifier): The trained model.
        data (torch_geometric.data.Data): The graph representation of the protein structure.
        T (float): The starting time (maximum noise level). Should match the training setup.
        t_min (float): The minimum time (initial noise level).
        steps (int): The number of denoising steps to perform.
        K (int): Number of amino acid classes (default: 21).
        verbose (bool): If True, prints detailed velocity and c factor debugging information.
        args: Arguments object with sampling parameters.

    Returns:
        tuple: (final_probabilities, predicted_sequence, trajectory_data, evaluation_metrics)
    """
    if verbose:
        print("="*80)
        print("TRAJECTORY SAMPLING: Detailed Analysis with Full Tracking")
        print("="*80)

    # Ensure model is on the correct device and in evaluation mode
    device = next(model.parameters()).device
    model.eval()

    # Prepare batched data
    from training.collate import collate_fn
    dummy_y = torch.zeros(1, K)
    dummy_mask = torch.ones(1, dtype=torch.bool)
    dummy_time = torch.tensor(0.0)  # Dummy time value
    batched_data, y_pad, mask_pad, time_batch = collate_fn([(data, dummy_y, dummy_mask, dummy_time)])
    batched_data = batched_data.to(device)

    # Handle virtual nodes
    total_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x_s.size(0)
    use_virtual_node = getattr(data, 'use_virtual_node', False)

    if use_virtual_node:
        N = total_nodes - 1
    else:
        N = total_nodes

    if N <= 0:
        raise ValueError(f"No real nodes found. Total nodes: {total_nodes}, use_virtual_node: {use_virtual_node}")

    print(f"Generating sequence with trajectory for {N} residues (use_virtual_node: {use_virtual_node})")

    # Start with Dirichlet noise
    dirichlet_concentration = args.dirichlet_concentration if args else 20.0
    dirichlet_dist = Dirichlet(dirichlet_concentration * torch.ones(K, device=device))
    x = dirichlet_dist.sample((1, N))  # Shape: [1, N, K]

    # Time steps - go from low t (noisy) to high t (clean)
    times = torch.linspace(t_min, T, steps, device=device)
    dt = (T - t_min) / (steps - 1)

    print(f"Starting reverse sampling from t={t_min:.1f} to t={T:.1f} in {steps} steps")

    # Initialize trajectory tracking
    trajectory_data = {
        'time_points': [],
        'positions': {}
    }

    # Initialize position data with detailed amino acid breakdown
    for pos in range(N):
        trajectory_data['positions'][pos] = {
            'time_points': [],
            'most_likely_aa': [],
            'probabilities': [],
            'detailed_breakdown': []  # store detailed AA breakdown
        }

    with torch.no_grad():
        time_steps = tqdm(enumerate(times), total=len(times), desc="Sampling with trajectory", disable=not verbose)
        for i, t_val in time_steps:
            t = torch.full((1,), t_val, device=device)

            # Store current state
            current_probs = x.squeeze(0).cpu().numpy()  # [N, K]
            current_most_likely = x.argmax(-1).squeeze(0).cpu().tolist()  # [N]
            current_max_probs = x.max(-1)[0].squeeze(0).cpu().tolist()  # [N]

            # Update progress bar with current step info
            current_entropy = -(x * torch.log(x + 1e-10)).sum(-1).mean().item()
            current_avg_max_prob = x.max(-1)[0].mean().item()
            time_steps.set_postfix({
                't': f'{t_val:.3f}',
                'entropy': f'{current_entropy:.4f}',
                'avg_max_prob': f'{current_avg_max_prob:.4f}'
            })

            # Skip velocity prediction on last step
            if i == len(times) - 1:
                break

            # UPDATED FOR POSITION PREDICTION: Get predicted target from model
            model_output = model(batched_data, t, x)
            protein_source = 'pdb'

            if protein_source == "unknown":
                print("Failed to extract the protein source")
                #raise Exception("Failed to extract the protein source")

            # Handle DSSP multitask output - model might return (sequence_logits, dssp_logits) tuple
            if isinstance(model_output, tuple):
                position_logits = model_output[0]  # Use only sequence logits for sampling
                if verbose and i == 0:  # Only print once to avoid spam
                    print("  Model returns tuple (sequence + DSSP), using sequence logits for sampling")
            else:
                position_logits = model_output

            # DEBUG: Analyze raw logits before temperature scaling
            if i % 5 == 0 or i == len(times) - 1:  # Every 5 steps or final step
                print(f"\n=== DEBUG STEP {i} (t={t_val:.3f}) ===")
                print(f"Raw logits shape: {position_logits.shape}")
                print(f"Raw logits stats - min: {position_logits.min().item():.4f}, max: {position_logits.max().item():.4f}")
                print(f"Raw logits stats - mean: {position_logits.mean().item():.4f}, std: {position_logits.std().item():.4f}")

                # DEBUG: Check actual RBF parameters being used by the model
                if i == 0:  # Only check on first step
                    print(f"=== MODEL RBF VERIFICATION ===")
                    print(f"Model type: {type(model)}")

                    # Check the batched_data structure more thoroughly
                    print(f"Batched data type: {type(batched_data)}")

                    # Check edge attributes specifically
                    for attr in ['edge_attr', 'edge_attr_3d', 'edge_distances', 'edge_index']:
                        if hasattr(batched_data, attr):
                            val = getattr(batched_data, attr)
                            if val is not None:
                                print(f"{attr} shape: {val.shape if hasattr(val, 'shape') else 'N/A'}")
                                if hasattr(val, 'min') and hasattr(val, 'max'):
                                    print(f"{attr} range: {val.min().item():.4f} - {val.max().item():.4f}")

                    # CRITICAL: Check which RBF file is actually being used
                    # Look for RBF manager in the data object
                    if hasattr(batched_data, '_rbf_manager'):
                        rbf_mgr = batched_data._rbf_manager
                        print(f"RBF Manager 3D params: min={rbf_mgr.rbf_3d_min}, max={rbf_mgr.rbf_3d_max}")
                        print(f"RBF 3D filename: {rbf_mgr.rbf_3d_filename}")
                        print(f"RBF cache directory: {rbf_mgr.cache_dir}")

                    # Also check if the data has been pre-processed with RBF features
                    if hasattr(batched_data, 'edge_attr'):
                        ea = batched_data.edge_attr
                        if ea is not None:
                            print(f"Edge attributes (RBF features) shape: {ea.shape}")
                            print(f"Edge attributes range: {ea.min().item():.4f} - {ea.max().item():.4f}")
                            print(f"Non-zero edge attributes: {(ea > 0).sum().item()}/{ea.numel()}")

                            # Check if the values look like RBF features (should be between 0 and 1)
                            if ea.max().item() > 1.0 or ea.min().item() < 0.0:
                                print(f"WARNING: Edge attributes outside [0,1] range - may not be RBF features")

                    print(f"=== END RBF VERIFICATION ===")

                # Analyze class distribution before temperature scaling
                raw_probs = torch.softmax(position_logits, dim=-1)
                if use_virtual_node:
                    raw_probs_real = raw_probs[:N, :]
                else:
                    raw_probs_real = raw_probs

                # Get dominant class for each position
                dominant_classes = torch.argmax(raw_probs_real, dim=-1)
                class_counts = torch.bincount(dominant_classes, minlength=21)
                class_percentages = (class_counts.float() / dominant_classes.numel() * 100)


            # DEBUG: Analyze raw logits before temperature scaling
            if i % 5 == 0 or i == len(times) - 1:  # Every 5 steps or final step
                print(f"\n=== DEBUG STEP {i} (t={t_val:.3f}) ===")
                print(f"Raw logits shape: {position_logits.shape}")
                print(f"Raw logits stats - min: {position_logits.min().item():.4f}, max: {position_logits.max().item():.4f}")
                print(f"Raw logits stats - mean: {position_logits.mean().item():.4f}, std: {position_logits.std().item():.4f}")

                # DEBUG: Check actual RBF parameters being used by the model
                if i == 0:  # Only check on first step
                    print(f"=== MODEL RBF VERIFICATION ===")
                    print(f"Model type: {type(model)}")

                    # Check the batched_data structure more thoroughly
                    print(f"Batched data type: {type(batched_data)}")

                    # Check edge attributes specifically
                    for attr in ['edge_attr', 'edge_attr_3d', 'edge_distances', 'edge_index']:
                        if hasattr(batched_data, attr):
                            val = getattr(batched_data, attr)
                            if val is not None:
                                print(f"{attr} shape: {val.shape if hasattr(val, 'shape') else 'N/A'}")
                                if hasattr(val, 'min') and hasattr(val, 'max'):
                                    print(f"{attr} range: {val.min().item():.4f} - {val.max().item():.4f}")

                    # CRITICAL: Check which RBF file is actually being used
                    # Look for RBF manager in the data object
                    if hasattr(batched_data, '_rbf_manager'):
                        rbf_mgr = batched_data._rbf_manager
                        print(f"RBF Manager 3D params: min={rbf_mgr.rbf_3d_min}, max={rbf_mgr.rbf_3d_max}")
                        print(f"RBF 3D filename: {rbf_mgr.rbf_3d_filename}")
                        print(f"RBF cache directory: {rbf_mgr.cache_dir}")

                    # Also check if the data has been pre-processed with RBF features
                    if hasattr(batched_data, 'edge_attr'):
                        ea = batched_data.edge_attr
                        if ea is not None:
                            print(f"Edge attributes (RBF features) shape: {ea.shape}")
                            print(f"Edge attributes range: {ea.min().item():.4f} - {ea.max().item():.4f}")
                            print(f"Non-zero edge attributes: {(ea > 0).sum().item()}/{ea.numel()}")

                            # Check if the values look like RBF features (should be between 0 and 1)
                            if ea.max().item() > 1.0 or ea.min().item() < 0.0:
                                print(f"WARNING: Edge attributes outside [0,1] range - may not be RBF features")

                    print(f"=== END RBF VERIFICATION ===")

                # Analyze class distribution before temperature scaling
                raw_probs = torch.softmax(position_logits, dim=-1)
                if use_virtual_node:
                    raw_probs_real = raw_probs[:N, :]
                else:
                    raw_probs_real = raw_probs

                # Get dominant class for each position
                dominant_classes = torch.argmax(raw_probs_real, dim=-1)
                class_counts = torch.bincount(dominant_classes, minlength=21)
                class_percentages = (class_counts.float() / dominant_classes.numel() * 100)


            # Convert to predicted target distribution
            # Apply time-dependent temperature if requested
            if args and getattr(args, 'time_as_temperature', False):
                # Temperature starts high (at t_min) and decreases as we approach t_max
                flow_temp = T - t_val + 0.1
            else:
                flow_temp = args.flow_temp if args else 1.0

            predicted_target = torch.softmax(position_logits/flow_temp, dim=-1)

            # Extract only real node predictions and ensure same shape as x
            if use_virtual_node:
                # position_logits has shape [total_nodes, K], need to slice out real nodes only
                # For single structure: real nodes are indices 0 to N-1, virtual node is at index N
                predicted_target_real = predicted_target[:N, :].unsqueeze(0)  # [N, K] -> [1, N, K]
            else:
                # If no virtual node, still need to add batch dimension to match x shape
                predicted_target_real = predicted_target.unsqueeze(0)  # [N, K] -> [1, N, K]

            # Compute analytical velocity using conditional flow
            cond_flow = model.cond_flow
            v_analytical = cond_flow.velocity(
                x,
                predicted_target_real,
                t,
                use_virtual_node=use_virtual_node,
                use_smoothed_targets=getattr(args, 'use_smoothed_targets', False),
                use_c_factor=getattr(args, 'use_c_factor', False)
            )

            # Record detailed trajectory data with velocity and flow components
            current_probs = x.squeeze(0).cpu().numpy()  # [N, K]
            current_most_likely = x.argmax(-1).squeeze(0).cpu().tolist()  # [N]
            current_max_probs = x.max(-1)[0].squeeze(0).cpu().tolist()  # [N]
            predicted_probs = predicted_target_real.squeeze(0).cpu().numpy()  # [N, K]
            velocities = v_analytical.squeeze(0).cpu().numpy()  # [N, K]

            # Update trajectory data with detailed breakdown
            trajectory_data['time_points'].append(float(t_val))
            for pos in range(N):
                trajectory_data['positions'][pos]['time_points'].append(float(t_val))
                trajectory_data['positions'][pos]['most_likely_aa'].append(current_most_likely[pos])
                trajectory_data['positions'][pos]['probabilities'].append(current_max_probs[pos])

                # Store detailed amino acid breakdown with velocity components
                aa_breakdown = {}
                aa_names = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
                for k, aa_name in enumerate(aa_names):
                    aa_breakdown[aa_name] = {
                        'current_prob': round(float(current_probs[pos, k]), 6),
                        'predicted_prob': round(float(predicted_probs[pos, k]), 6) if k < predicted_probs.shape[1] else 0.0,
                        'velocity_component': round(float(velocities[pos, k]), 6) if k < velocities.shape[1] else 0.0,
                        'c_factor_component': 0.0  # Placeholder - would need to compute c_factor separately
                    }
                trajectory_data['positions'][pos]['detailed_breakdown'].append(aa_breakdown)

            if verbose and (i + 1) % 10 == 0:
                current_entropy = -(x * torch.log(x + 1e-10)).sum(-1).mean().item()
                current_avg_max_prob = x.max(-1)[0].mean().item()
                print(f"  Step {i+1}/{steps}: t={t_val:.3f}, entropy={current_entropy:.4f}, avg_max_prob={current_avg_max_prob:.4f}")

            # Use analytical velocity for integration
            v_processed = v_analytical

            # Update sequence using Euler step with proper simplex projection
            x_new = x + dt * v_processed
            x_new = simplex_proj(x_new)  # Use proper simplex projection instead of clamp + normalize
            x = x_new

    final_probabilities = x.squeeze(0)
    predicted_sequence = final_probabilities.argmax(-1).tolist()

    # Compute evaluation metrics if ground truth is available
    evaluation_metrics = None
    ground_truth = getattr(data, 'y', None)
    if ground_truth is not None:
        # Extract ground truth for current structure (handle padding)
        if use_virtual_node and ground_truth.size(0) > N:
            ground_truth_clean = ground_truth[:N]
        else:
            ground_truth_clean = ground_truth

        evaluation_metrics = compute_sampling_metrics(
            final_probabilities, ground_truth_clean, data, model, args, device, use_virtual_node, K
        )

        if verbose:
            print(f"\nEVALUATION METRICS:")
            print(f"  Accuracy: {evaluation_metrics['accuracy']:.4f}")
            print(f"  CCE (hard labels): {evaluation_metrics['cce_loss_hard']:.4f}")
            print(f"  CCE (smooth labels): {evaluation_metrics['cce_loss_smooth']:.4f}")
            print(f"  Valid positions: {evaluation_metrics.get('valid_positions', 'N/A')}/{evaluation_metrics.get('total_positions', 'N/A')} ({evaluation_metrics.get('fraction_valid', 0.0)*100:.1f}%)")
            print(f"  Unknown positions excluded: {evaluation_metrics.get('unknown_positions', 'N/A')}")

    return final_probabilities, predicted_sequence, trajectory_data, evaluation_metrics


def main():
    """Main entry point for protein sequence sampling."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Process arguments to handle interdependent logic (e.g., use_smoothed_labels -> use_smoothed_targets)
    from training.sample_utils import process_sampling_args
    args = process_sampling_args(args)

    # Disable probability saving if explicitly requested
    if args.no_probabilities:
        args.save_probabilities = False

    print("="*60)
    print("PROTEIN SEQUENCE SAMPLING WITH DIRICHLET FLOW MATCHING")
    print("="*60)

    # Auto-enable detailed JSON for small number of proteins
    if not args.sample_all or (args.max_structures and args.max_structures < 4):
        if not args.detailed_json:
            pass
            #print("Automatically enabling detailed JSON output (fewer than 4 proteins)")
            #args.detailed_json = True

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and extract dataset parameters
    from training.sample_utils import load_model_distributed

    model, dataset_params = load_model_distributed(args.model_path, device, args)

    # Resolve graph builder parameters BEFORE any sampling mode
    # This ensures all modes use the same resolved parameters
    print("\n" + "="*60)
    print("DATASET PARAMETER RESOLUTION")
    print("="*60)

    # Extract dataset parameters with fallbacks and command line override capability
    split_json = args.split_json or dataset_params.get('split_json', '../datasets/cath-4.2/chain_set_splits.json')
    map_pkl = args.map_pkl or dataset_params.get('map_pkl', '../datasets/cath-4.2/chain_set_map_with_b_factors.pkl')
    use_virtual_node = dataset_params.get('use_virtual_node', True)
    max_length = getattr(args, 'max_length', None) or dataset_params.get('max_length')
    use_graph_builder = getattr(args, 'use_graph_builder', dataset_params.get('use_graph_builder', True))

    print(f"split_json: {split_json} (source: {'command line args' if args.split_json else 'checkpoint'})")
    print(f"map_pkl: {map_pkl} (source: {'command line args' if args.map_pkl else 'checkpoint'})")
    print(f"use_virtual_node: {use_virtual_node} (source: checkpoint)")
    print(f"max_length: {max_length} (source: {'not specified' if max_length is None else 'checkpoint'})")
    print(f"use_graph_builder: {use_graph_builder} (source: default)")
    print(f"\nGraph builder parameter resolution:")

    def resolve_param(param_name, checkpoint_value, args_value, param_type="parameter"):
        if args_value is not None:
            print(f"  {param_name}: {args_value} (source: command line)")
            return args_value
        elif checkpoint_value is not None:
            print(f"  {param_name}: {checkpoint_value} (source: checkpoint)")
            return checkpoint_value
        else:
            raise ValueError(f"Graph builder {param_type} '{param_name}' not found in checkpoint and not provided via command line. "
                           f"This parameter is required to match the training configuration.")

    print(f"\nResolving graph builder parameters:")

    # Handle max_edge_dist vs k-neighbors logic BEFORE resolving individual parameters
    max_edge_dist_from_checkpoint = dataset_params.get('max_edge_dist')
    if max_edge_dist_from_checkpoint is not None:
        print(f"  max_edge_dist found in checkpoint: {max_edge_dist_from_checkpoint}")
        print(f"  Setting k_neighbors, k_farthest, k_random to 0 (distance-based edges will be used)")
        # Override args to ensure k-neighbor parameters are 0 when max_edge_dist is used
        args.k_neighbors = 0
        args.k_farthest = 0
        args.k_random = 0

    k_neighbors = resolve_param('k_neighbors', dataset_params.get('k_neighbors'), args.k_neighbors)
    k_farthest = resolve_param('k_farthest', dataset_params.get('k_farthest'), args.k_farthest)
    k_random = resolve_param('k_random', dataset_params.get('k_random'), args.k_random)
    max_edge_dist = resolve_param('max_edge_dist', dataset_params.get('max_edge_dist'), getattr(args, 'max_edge_dist', None))
    num_rbf_3d = resolve_param('num_rbf_3d', dataset_params.get('num_rbf_3d'), args.num_rbf_3d)
    num_rbf_seq = resolve_param('num_rbf_seq', dataset_params.get('num_rbf_seq'), args.num_rbf_seq)

    # Extract RBF distance parameters from checkpoint
    rbf_3d_min = resolve_param('rbf_3d_min', dataset_params.get('rbf_3d_min'), getattr(args, 'rbf_3d_min', None))
    rbf_3d_max = resolve_param('rbf_3d_max', dataset_params.get('rbf_3d_max'), getattr(args, 'rbf_3d_max', None))
    rbf_3d_spacing = resolve_param('rbf_3d_spacing', dataset_params.get('rbf_3d_spacing'), getattr(args, 'rbf_3d_spacing', None))

    # Resolve no_source_indicator with default False if not found
    no_source_indicator_checkpoint = dataset_params.get('no_source_indicator')
    no_source_indicator_args = getattr(args, 'no_source_indicator', None)
    if no_source_indicator_args is not None:
        no_source_indicator = no_source_indicator_args
        print(f"  no_source_indicator: {no_source_indicator} (source: command line)")
    elif no_source_indicator_checkpoint is not None:
        no_source_indicator = no_source_indicator_checkpoint
        print(f"  no_source_indicator: {no_source_indicator} (source: checkpoint)")
    else:
        no_source_indicator = False  # Default value
        print(f"  no_source_indicator: {no_source_indicator} (source: default)")

    # Handle use_virtual_node specially since it's a boolean with different argument logic
    if args.no_virtual_node:
        final_use_virtual = False
        print(f"  use_virtual_node: {final_use_virtual} (source: command line --no_virtual_node)")
    elif args.use_virtual_node:
        final_use_virtual = True
        print(f"  use_virtual_node: {final_use_virtual} (source: command line --use_virtual_node)")
    elif use_virtual_node is not None:
        final_use_virtual = use_virtual_node
        print(f"  use_virtual_node: {final_use_virtual} (source: checkpoint/inference)")
    else:
        raise ValueError("use_virtual_node not found in checkpoint and not specified via command line flags. "
                       "Use --use_virtual_node or --no_virtual_node to specify, or ensure the checkpoint contains this information.")

    # Prepare graph builder parameters (used by all sampling modes)
    graph_builder_kwargs = {
        'k': k_neighbors,
        'k_farthest': k_farthest,
        'k_random': k_random,
        'max_edge_dist': max_edge_dist,  # Use resolved max_edge_dist
        'num_rbf_3d': num_rbf_3d,
        'num_rbf_seq': num_rbf_seq,
        'use_virtual_node': final_use_virtual,
        'no_source_indicator': no_source_indicator,
        'verbose': args.verbose,
        # Include RBF distance parameters from checkpoint
        'rbf_3d_min': rbf_3d_min,
        'rbf_3d_max': rbf_3d_max,
        'rbf_3d_spacing': rbf_3d_spacing
    }

    # Handle direct PDB input mode (--pdb_input argument)
    if args.pdb_input is not None:
        print("\n" + "="*60)
        print("DIRECT PDB INPUT MODE")
        print("="*60)
        print(f"PDB Input: {args.pdb_input}")

        # First try to find it in the dataset (if available) for faster processing
        dataset_entry = None
        if not os.path.exists(args.pdb_input):  # Only try dataset lookup for non-file inputs
            try:
                print("Attempting to find structure in dataset first...")

                # Load dataset parameters first to check if we can do dataset lookup
                split_json = dataset_params.get('split_json') or args.split_json
                map_pkl = dataset_params.get('map_pkl') or args.map_pkl

                if split_json and map_pkl and os.path.exists(split_json) and os.path.exists(map_pkl):
                    # Try to find the structure in dataset
                    from data.cath_dataset import CathDataset

                    # Create minimal dataset to check if structure exists
                    try:
                        temp_dataset = CathDataset(
                            split_json=split_json,
                            map_pkl=map_pkl,
                            split=args.split,
                            graph_builder_kwargs={'use_virtual_node': True},  # Minimal setup
                            time_sampling_strategy="uniform",
                            t_min=0,
                            t_max=args.T,
                            alpha_range=0
                        )

                        # Look for the structure in the dataset
                        target_name = args.pdb_input.upper()  # Convert to standard format
                        found_idx = None

                        for idx, entry in enumerate(temp_dataset.entries):
                            if entry['name'].upper() == target_name:
                                found_idx = idx
                                break

                        if found_idx is not None:
                            print(f"Found {target_name} in dataset at index {found_idx}, using dataset version")
                            dataset_entry = temp_dataset.entries[found_idx]
                        else:
                            print(f"Structure {target_name} not found in dataset, will download from PDB")

                    except Exception as e:
                        print(f"Dataset lookup failed ({e}), will download from PDB")

                else:
                    print("Dataset files not available, will download from PDB")

            except Exception as e:
                print(f"Error during dataset lookup: {e}")
                print("Falling back to direct PDB processing...")

        # Import enhanced input processing functions
        from training.sample_utils import (
            CustomInputDataset,
            process_input_specification,
        )

        temp_files = []
        try:
            if dataset_entry is not None:
                # Use dataset entry directly
                print("Using structure from dataset...")
                entry = dataset_entry
            else:
                # Process the input specification (download/parse files)
                print("Processing direct PDB input...")
                entry, temp_files = process_input_specification(args.pdb_input)
            print(f"Structure: {entry['name']}")
            print(f"Sequence length: {len(entry['seq'])}")
            print(f"Source: {entry['source']}")

            # Build graph using existing GraphBuilder
            from data.graph_builder import GraphBuilder

            # Use resolved graph builder parameters from checkpoint
            print("Building graph...")
            builder = GraphBuilder(**graph_builder_kwargs)  # Use all resolved parameters
            graph_data = builder.build_from_dict(entry, time_param=0.0)
            graph_data = graph_data.to(device)

            print(f"Graph: {graph_data.x_s.shape[0]} nodes, {graph_data.edge_index.shape[1]} edges")

            # Create fake dataset for compatibility
            fake_dataset = CustomInputDataset(graph_data, entry)

            # Sample using existing function
            print("Sampling sequence...")
            final_probabilities, predicted_sequence, eval_metrics = sample_chain(
                model, graph_data, fake_dataset,
                structure_idx=0,
                T=args.T,
                t_min=args.t_min,
                steps=args.steps,
                K=21,
                verbose=args.verbose,
                args=args
            )

            if predicted_sequence is None:
                print("Sampling failed")
                return 1

            # Convert indices to amino acids
            predicted_aa = []
            for idx in predicted_sequence:
                if 0 <= idx < len(IDX_TO_AA):
                    predicted_aa.append(IDX_TO_AA[idx])
                else:
                    print(f"Warning: Predicted index {idx} out of bounds, using 'XXX'")
                    predicted_aa.append('XXX')

            # Create predicted sequence string
            predicted_seq_str = ''.join([THREE_TO_ONE.get(aa, 'X') for aa in predicted_aa])

            # Print results
            print("\nRESULTS:")
            print("="*60)
            print(f"Structure: {entry['name']}")
            print(f"Length: {len(predicted_sequence)} residues")
            print(f"Predicted sequence: {predicted_seq_str}")

            # Save results
            print("Saving results...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create descriptive output name
            input_name = args.pdb_input.replace('/', '_').replace('.', '_')
            if len(args.pdb_input) == 4 and args.pdb_input.isalnum():
                input_name = args.pdb_input.upper()
            elif '.' in args.pdb_input and len(args.pdb_input.split('.')[0]) == 4:
                input_name = args.pdb_input.upper().replace('.', '_')

            output_prefix = f"pdb_{input_name}_{timestamp}"

            # Extract true sequence from PDB entry
            true_sequence_str = entry['seq']  # Original sequence from PDB (single-letter codes)

            # Convert single-letter codes to indices via three-letter codes
            true_indices = []
            for aa_single in true_sequence_str:
                aa_triple = SINGLE_TO_TRIPLE.get(aa_single, 'XXX')  # Convert single to triple
                aa_idx = AA_TO_IDX.get(aa_triple, 20)  # Convert triple to index
                true_indices.append(aa_idx)

            # Calculate accuracy if sequences are same length
            accuracy = None
            if len(predicted_sequence) == len(true_indices):
                matches = sum(1 for pred, true in zip(predicted_sequence, true_indices) if pred == true)
                accuracy = (matches / len(predicted_sequence)) * 100.0
            else:
                print(f"Warning: Length mismatch - predicted: {len(predicted_sequence)}, true: {len(true_indices)}")

            # Format results for saving
            formatted_results = [{
                'structure_idx': 0,
                'structure_name': entry['name'],
                'pdb_input_spec': args.pdb_input,
                'length': len(predicted_sequence),
                'predicted_indices': predicted_sequence,
                'predicted_aa': predicted_aa,
                'predicted_sequence': predicted_seq_str,
                'true_indices': true_indices,
                'true_sequence': true_sequence_str,
                'accuracy': accuracy,
                'final_probabilities': final_probabilities.cpu().numpy() if args.save_probabilities else None,
                'source': entry['source'],
                'eval_metrics': eval_metrics
            }]

            # Save using existing function
            file_info = save_results_to_files(
                formatted_results,
                output_prefix,
                args.output_dir,  # Add missing output_dir parameter
                model_name=os.path.basename(args.model_path).replace('.pt', ''),
                split='custom',
                steps=args.steps,
                T=args.T
            )

            print("Results saved:")
            for key, path in file_info.items():
                if path:
                    print(f"  {key}: {path}")

            print("\nDirect PDB input sampling completed successfully!")
            return 0

        except Exception as e:
            print(f"Error in direct PDB input mode: {e}")
            import traceback
            if args.verbose:
                traceback.print_exc()
            return 1

        finally:
            # Cleanup temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        if args.verbose:
                            print(f"Cleaned up: {temp_file}")
                except:
                    pass

    # Standard dataset mode continues below...
    # Use dataset parameters from checkpoint if available, otherwise use args
    split_json = dataset_params.get('split_json') or args.split_json
    map_pkl = dataset_params.get('map_pkl') or args.map_pkl
    use_virtual_node = dataset_params.get('use_virtual_node', True)

    # Load dataset with parameters from checkpoint
    print(f"\nLoading dataset from: {split_json}")
    print(f"Using data mapping from: {map_pkl}")

    # Check if we need to optimize dataset loading with a protein list filter
    protein_list_filter = None
    if args.protein_list:
        # Load protein list early to use as a filter for efficient dataset loading
        from training.sample_utils import parse_protein_list_from_file
        try:
            protein_list_filter = parse_protein_list_from_file(args.protein_list)
            print(f"Will use protein list filter to load only {len(protein_list_filter)} proteins from dataset")
        except Exception as e:
            print(f"Warning: Could not load protein list for filtering: {e}")
            print("Falling back to loading full dataset and filtering later")

    # Check if we're sampling a single protein by name to optimize loading
    use_single_protein_dataset = (
        args.protein_name is not None and
        not args.sample_all and
        not args.protein_names and
        not args.protein_indices and
        not args.protein_list
    )

    if use_single_protein_dataset:
        print(f"Using SingleProteinDataset for efficient loading of '{args.protein_name}'")
        from data.single_protein_dataset import SingleProteinDataset

        try:
            dataset = SingleProteinDataset(
                split_json=split_json,
                map_pkl=map_pkl,
                protein_name=args.protein_name,
                split=args.split,
                graph_builder_kwargs=graph_builder_kwargs,
                time_sampling_strategy="uniform",
                t_min=0,
                t_max=args.T,
                alpha_range=0
            )
            print(f"Successfully loaded single protein dataset: {len(dataset)} entry")
        except Exception as e:
            print(f"Failed to use SingleProteinDataset: {e}")
            print("Falling back to full CathDataset...")
            use_single_protein_dataset = False

    if not use_single_protein_dataset:
        from data.cath_dataset import CathDataset

        try:
            dataset = CathDataset(
                split_json=split_json,
                map_pkl=map_pkl,
                split=args.split,
                graph_builder_kwargs=graph_builder_kwargs,
                time_sampling_strategy="uniform",
                t_min=0,
                t_max=args.T,  # Use args.T instead of args.t_max
                alpha_range=0,
                protein_list_filter=protein_list_filter
            )
        except FileNotFoundError as e:
            if 'chain_set_splits.json' in str(e):
                # Try alternative paths for chain_set_splits.json
                alt_paths = [
                    'datasets/cath-4.2/chain_set_splits.json',
                    './datasets/cath-4.2/chain_set_splits.json',
                    '../datasets/cath-4.2/chain_set_splits.json'
                ]

                dataset = None
                for alt_path in alt_paths:
                    try:
                        print(f"Trying alternative path: {alt_path}")
                        alt_map_pkl = map_pkl.replace('../datasets/', 'datasets/').replace('./datasets/', 'datasets/')
                        dataset = CathDataset(
                            split_json=alt_path,
                            map_pkl=alt_map_pkl,
                            split=args.split,
                            graph_builder_kwargs=graph_builder_kwargs,
                            time_sampling_strategy="uniform",
                            t_min=0,
                            t_max=args.T,
                            alpha_range=0,
                            protein_list_filter=protein_list_filter
                        )
                        print(f"Successfully loaded dataset from: {alt_path}")
                        break
                    except FileNotFoundError:
                        continue

                if dataset is None:
                    print(f"Error: Could not find chain_set_splits.json in any of the expected locations:")
                    for path in [split_json] + alt_paths:
                        print(f"  - {path}")
                    raise FileNotFoundError(f"chain_set_splits.json not found. Please check dataset paths.")
            else:
                raise e

    print(f"Loaded {len(dataset)} entries for {args.split} split")
    print(f"Dataset loaded: {len(dataset)} structures in {args.split} split")
    print("="*60)

    # Import utility functions for protein sampling
    from training.sample_utils import resolve_protein_sampling_mode

    # Determine sampling mode and target structures using new unified approach
    try:
        indices, sampling_description = resolve_protein_sampling_mode(args, dataset)
        print(f"Sampling mode: {sampling_description}")
        print(f"Indices to sample: {indices[:10]}{'...' if len(indices) > 10 else ''}")
    except Exception as e:
        print(f"Error resolving protein sampling mode: {e}")
        return

    # Determine if this is single protein sampling or multiple protein sampling
    is_single_protein = len(indices) == 1 and not (args.protein_list or args.protein_names or args.protein_indices or args.sample_all)

    if not is_single_protein:
        # Multiple protein sampling
        print(f"Sampling sequences for {len(indices)} structures...")

    # Import all sampling functions at the top to avoid UnboundLocalError
    from training.sample_utils import (
        sample_multiple_proteins,
        sample_multiple_proteins_batched,
        sample_multiple_proteins_with_ensemble,
        sample_multiple_proteins_with_trajectory,
    )

    # Determine if we're in multi-protein mode
    if len(indices) > 1:
        # Multiple protein sampling - FORCE BATCHED SAMPLING ONLY (or if force_batch flag is set)
        force_batched = getattr(args, 'force_batch', True)  # Default to True for multiple proteins

        if force_batched:
            print(f"MULTI-PROTEIN BATCHED SAMPLING MODE: {len(indices)} proteins")
            print("Skipping individual sampling - using batched sampling only for efficiency")
            print("="*60)
        else:
            print(f"MULTI-PROTEIN SAMPLING MODE: {len(indices)} proteins")
            print("="*60)

        # Check if ensemble sampling is requested for multi-protein mode
        if args.ensemble_size > 1:
            print(f"ENSEMBLE MODE: {args.ensemble_size} replicas each")

            # Use ensemble multi-protein sampling
            results = sample_multiple_proteins_with_ensemble(
                model=model,
                dataset=dataset,
                indices=indices,
                steps=args.steps,
                T=args.T,
                t_min=args.t_min,
                K=21,
                ensemble_size=args.ensemble_size,
                consensus_strength=args.ensemble_consensus_strength,
                structure_noise_mag_std=args.structure_noise_mag_std or 0.0,
                uncertainty_struct_noise_scaling=args.uncertainty_struct_noise_scaling,
                batch_size=args.batch_size or 32,
                args=args
            )
            structure_names = [r.get('structure_name', f"structure_{r['structure_idx']}") for r in results]

        elif force_batched:
            # Force batched sampling for all multi-protein cases
            batch_size = args.batch_size or 128  # Default to reasonable batch size
            print(f"Using batched sampling with batch_size={batch_size}")

            results = sample_multiple_proteins_batched(
                model, dataset, indices=indices, steps=args.steps, T=args.T, t_min=args.t_min, K=21,
                save_probabilities=args.save_probabilities, integration_method=args.integration_method,
                batch_size=batch_size, args=args
            )
            structure_names = [r.get('structure_name', f"structure_{r['structure_idx']}") for r in results]

        else:
            # Legacy multi-protein sampling with multiple options (kept for backwards compatibility)
            if args.batch_size and args.batch_size > 1:
                # Use batched sampling for efficiency
                print(f"Using batched sampling with batch_size={args.batch_size}")
                results = sample_multiple_proteins_batched(
                    model, dataset, indices=indices, steps=args.steps, T=args.T, t_min=args.t_min, K=21,
                    save_probabilities=args.save_probabilities, integration_method=args.integration_method,
                    batch_size=args.batch_size, args=args
                )
                structure_names = [r.get('structure_name', f"structure_{r['structure_idx']}") for r in results]
            elif args.detailed_json:
                # Use trajectory tracking for small numbers of structures
                results, structure_names = sample_multiple_proteins_with_trajectory(
                    model, dataset, indices=indices, steps=args.steps, T=args.T, t_min=args.t_min, K=21,
                    integration_method=args.integration_method, rtol=args.rtol, atol=args.atol,
                    output_dir=args.output_dir, output_prefix=args.output_prefix, args=args
                )
            else:
                # Use serial processing
                results = sample_multiple_proteins(
                    model, dataset, indices=indices, steps=args.steps, T=args.T, K=21,
                    save_probabilities=args.save_probabilities, integration_method=args.integration_method,
                    rtol=args.rtol, atol=args.atol, args=args
                )
                structure_names = [r.get('structure_name', f"structure_{r['structure_idx']}") for r in results]

        # Save results
        save_results_to_files(
            results, args.output_prefix, args.output_dir,
            model_name=args.model_name if hasattr(args, 'model_name') else None,
            split=args.split, steps=args.steps, T=args.T
        )

    else:
        # Single protein sampling
        actual_structure_idx = indices[0]

        if actual_structure_idx >= len(dataset):
            print(f"Error: Structure index {actual_structure_idx} out of range (max: {len(dataset)-1})")
            exit(1)

        print(f"Getting structure {actual_structure_idx} from {args.split} split...")
        data, y_true, mask, time_value, dssp_targets = dataset[actual_structure_idx]  # Unpack 5 values (includes DSSP)


        # Extract structure name/identifier
        structure_name = getattr(data, 'name', f'structure_{actual_structure_idx}')
        protein_name = structure_name  # Use structure name as protein name
        protein_source = getattr(data, 'source', 'unknown')
        if protein_source == "unknown":
            raise Exception("Failed to extract the protein source")
        print(f"Structure name: {structure_name}")
        print(f"Structure source: {protein_source}")
        if args.protein_name:
            print(f"Selected protein: {structure_name} (dataset index: {actual_structure_idx})")


        # Initialize ground truth variables to avoid scoping issues
        true_sequence = None
        true_aa = None
        accuracy = None

        # Get number of samples per protein
        num_samples = getattr(args, 'num_sample_per_protein', 1)

        if num_samples > 1:
            print(f"Sampling {num_samples} sequences for protein {structure_name}...")

            # Multiple samples for single protein
            all_results = []
            for sample_idx in range(num_samples):
                print(f"  Generating sample {sample_idx+1}/{num_samples}...")

                # Sample the sequence based on mode
                if args.detailed_json:
                    final_probabilities, predicted_sequence, trajectory_data, eval_metrics = sample_chain_with_trajectory(
                        model, data, T=args.T, t_min=args.t_min, steps=args.steps, K=21, verbose=args.verbose, args=args
                    )
                else:
                    # Standard sampling
                    final_probabilities, predicted_sequence, eval_metrics = sample_chain(
                        model, data, dataset, structure_idx=actual_structure_idx, T=args.T, t_min=args.t_min, steps=args.steps, K=21, verbose=args.verbose, args=args
                    )

                # Convert indices to amino acids
                predicted_aa = []
                for idx in predicted_sequence:
                    if 0 <= idx < len(IDX_TO_AA):
                        predicted_aa.append(IDX_TO_AA[idx])
                    else:
                        print(f"Warning: Predicted index {idx} out of bounds, using 'XXX'")
                        predicted_aa.append('XXX')

                # Calculate accuracy if ground truth available
                accuracy = None
                true_sequence = None
                true_aa = None

                if y_true is not None:
                    # If y_true is all 20s (XXX), use the filtered_seq from data object instead
                    true_sequence = y_true.argmax(-1).tolist()

                    # Check if y_true is corrupted (all unknowns)
                    if all(idx == 20 for idx in true_sequence) and hasattr(data, 'filtered_seq'):
                        print("WARNING: y_true contains all unknown amino acids, using data.filtered_seq instead")

                        # Convert filtered_seq to indices
                        true_sequence = []
                        for aa_char in data.filtered_seq:
                            if aa_char in SINGLE_TO_TRIPLE:
                                aa3 = SINGLE_TO_TRIPLE[aa_char]
                                if aa3 in AA_TO_IDX:
                                    true_sequence.append(AA_TO_IDX[aa3])
                                else:
                                    true_sequence.append(20)  # Unknown
                            else:
                                true_sequence.append(20)  # Unknown

                    # Bounds checking for true indices
                    true_aa = []
                    for idx in true_sequence:
                        if 0 <= idx < len(IDX_TO_AA):
                            true_aa.append(IDX_TO_AA[idx])
                        else:
                            true_aa.append('XXX')

                    # Calculate accuracy
                    correct = sum(p == t for p, t in zip(predicted_sequence, true_sequence))
                    accuracy = correct / len(predicted_sequence) * 100

                print(f"\n  SAMPLE {sample_idx+1} RESULTS:")
                print("  " + "="*58)
                print(f"  Predicted sequence ({len(predicted_sequence)} residues):")
                print("  Indices:", predicted_sequence)
                print("  AA codes:", predicted_aa)
                print("  Sequence:", ''.join([THREE_TO_ONE[aa] for aa in predicted_aa]))

                if true_sequence is not None:
                    print(f"  Ground truth sequence ({len(true_sequence)} residues):")
                    print("  Indices:", true_sequence)
                    print("  AA codes:", true_aa)
                    print("  Sequence:", ''.join([THREE_TO_ONE[aa] for aa in true_aa]))
                    print(f"  Accuracy: {accuracy:.2f}%")

                # Store result
                result_dict = {
                    'structure_idx': actual_structure_idx,
                    'structure_name': structure_name,
                    'sample_idx': sample_idx,
                    'protein_name': protein_name,
                    'protein_source': protein_source,
                    'length': len(predicted_sequence),
                    'predicted_indices': predicted_sequence,
                    'predicted_aa': predicted_aa,
                    'predicted_sequence': ''.join([THREE_TO_ONE[aa] for aa in predicted_aa]),
                    'true_indices': true_sequence if y_true is not None else None,
                    'true_aa': true_aa if y_true is not None and true_aa else None,
                    'true_sequence': ''.join([THREE_TO_ONE[aa] for aa in true_aa]) if y_true is not None and true_aa else None,
                    'accuracy': accuracy,
                    'final_probabilities': final_probabilities.cpu().numpy()
                }

                # Add trajectory data if it was generated
                if args.detailed_json and 'trajectory_data' in locals():
                    result_dict['trajectory_data'] = trajectory_data

                all_results.append(result_dict)

            # Save multiple results
            print(f"\nSaving {num_samples} samples for protein {structure_name}...")
            model_name = os.path.basename(args.model_path).replace('.pt', '')
            output_prefix = f"{protein_name.replace('.', '_')}_multi" if args.protein_name else f"structure_{actual_structure_idx}_multi"

            file_info = save_results_to_files(
                all_results, output_prefix, args.output_dir,
                model_name=model_name, split=args.split, steps=args.steps, T=args.T
            )

            print(f"Files saved:")
            if 'sample_files' in file_info:
                for sample_idx, sample_file_info in file_info['sample_files'].items():
                    print(f"  Sample {sample_idx+1}:")
                    print(f"    Sequences: {sample_file_info['sequences_file']}")
                    print(f"    Probabilities: {sample_file_info['probabilities_file']}")
                    print(f"    Metadata: {sample_file_info['metadata_file']}")
            else:
                print(f"  Sequences: {file_info['sequences_file']}")
                print(f"  Probabilities: {file_info['probabilities_file']}")
                print(f"  Metadata: {file_info['metadata_file']}")

            results = all_results
            structure_names = [structure_name] * num_samples

        else:
            # Single sample (original behavior)

            # Check if ensemble sampling is requested
            if args.ensemble_size > 1:
                print(f"ENSEMBLE SAMPLING MODE: {args.ensemble_size} replicas")
                print("="*60)

                # Import ensemble functions
                from training.sample_utils import (
                    create_structural_ensemble,
                    sample_with_ensemble_consensus,
                )

                # Get the entry for this structure from dataset
                if hasattr(dataset, 'entries'):
                    # CathDataset has entries list
                    original_entry = dataset.entries[actual_structure_idx]
                elif hasattr(dataset, 'protein_entry'):
                    # SingleProteinDataset has single protein_entry
                    original_entry = dataset.protein_entry
                else:
                    raise AttributeError(f"Unknown dataset type: {type(dataset)}")

                # Create ensemble from the entry
                batched_ensemble = create_structural_ensemble(
                    original_entry,
                    ensemble_size=args.ensemble_size,
                    structure_noise_mag_std=args.structure_noise_mag_std or 0.0,
                    uncertainty_struct_noise_scaling=args.uncertainty_struct_noise_scaling,
                    device=device,
                    args=args,
                    dataset_params=dataset_params
                )

                # Sample with ensemble consensus
                predicted_sequence = sample_with_ensemble_consensus(
                    model=model,
                    batched_ensemble=batched_ensemble,
                    T=args.T,
                    t_min=args.t_min,
                    steps=args.steps,
                    K=21,
                    consensus_strength=args.ensemble_consensus_strength,
                    device=device,
                    use_virtual_node=dataset_params.get('use_virtual_node', False),
                    args=args
                )

                # Create dummy probabilities for compatibility (we only return consensus sequence)
                final_probabilities = torch.zeros(len(predicted_sequence), 21)
                for i, idx in enumerate(predicted_sequence):
                    final_probabilities[i, idx] = 1.0  # One-hot for consensus

                eval_metrics = {}  # No evaluation metrics for ensemble mode yet

            else:
                # Standard single-structure sampling
                if args.detailed_json:
                    print("Generating detailed JSON output with time-step information")

                    final_probabilities, predicted_sequence, trajectory_data, eval_metrics = sample_chain_with_trajectory(
                        model, data, T=args.T, t_min=args.t_min, steps=args.steps, K=21, verbose=args.verbose, args=args
                    )

                else:
                    # Standard sampling
                    final_probabilities, predicted_sequence, eval_metrics = sample_chain(
                        model, data, dataset, structure_idx=actual_structure_idx, T=args.T, t_min=args.t_min, steps=args.steps, K=21, verbose=args.verbose, args=args
                    )

            # Convert indices to amino acids
            predicted_aa = []
            for idx in predicted_sequence:
                if 0 <= idx < len(IDX_TO_AA):
                    predicted_aa.append(IDX_TO_AA[idx])
                else:
                    print(f"Warning: Predicted index {idx} out of bounds, using 'XXX'")
                    predicted_aa.append('XXX')

            print("\nRESULTS:")
            print("="*60)
            print(f"Predicted sequence ({len(predicted_sequence)} residues):")
            print("Indices:", predicted_sequence)
            print("AA codes:", predicted_aa)
            print("Sequence:", ''.join([THREE_TO_ONE[aa] for aa in predicted_aa]))  # Single letter codes

            # Initialize ground truth variables to avoid scoping issues
            true_sequence = None
            true_aa = None
            accuracy = None

            # If we have ground truth, compare
            if y_true is not None:
                # If y_true is all 20s (XXX), use the filtered_seq from data object instead
                true_sequence = y_true.argmax(-1).tolist()

                # Check if y_true is corrupted (all unknowns)
                if all(idx == 20 for idx in true_sequence) and hasattr(data, 'filtered_seq'):
                    print("WARNING: y_true contains all unknown amino acids, using data.filtered_seq instead")

                    # Convert filtered_seq to indices
                    true_sequence = []
                    for aa_char in data.filtered_seq:
                        if aa_char in SINGLE_TO_TRIPLE:
                            aa3 = SINGLE_TO_TRIPLE[aa_char]
                            if aa3 in AA_TO_IDX:
                                true_sequence.append(AA_TO_IDX[aa3])
                            else:
                                true_sequence.append(20)  # Unknown
                        else:
                            true_sequence.append(20)  # Unknown

                # Bounds checking for true indices
                true_aa = []
                for idx in true_sequence:
                    if 0 <= idx < len(IDX_TO_AA):
                        true_aa.append(IDX_TO_AA[idx])
                    else:
                        print(f"Warning: True index {idx} out of bounds, using 'XXX'")
                        true_aa.append('XXX')

                print(f"\nGround truth sequence ({len(true_sequence)} residues):")
                print("Indices:", true_sequence)
                print("AA codes:", true_aa)
                print("Sequence:", ''.join([THREE_TO_ONE[aa] for aa in true_aa]))

                # Calculate accuracy
                correct = sum(p == t for p, t in zip(predicted_sequence, true_sequence))
                accuracy = correct / len(predicted_sequence) * 100
                print(f"\nAccuracy: {correct}/{len(predicted_sequence)} = {accuracy:.1f}%")

            # Save single structure results
            print(f"\nSaving single structure results...")
            result_dict = {
                'structure_idx': actual_structure_idx,
                'structure_name': structure_name,
                'protein_name': structure_name,
                'protein_source': protein_source,
                'length': len(predicted_sequence),
                'predicted_indices': predicted_sequence,
                'predicted_aa': predicted_aa,
                'predicted_sequence': ''.join([THREE_TO_ONE[aa] for aa in predicted_aa]),
                'true_indices': true_sequence if y_true is not None else None,
                'true_aa': true_aa if y_true is not None else None,
                'true_sequence': ''.join([THREE_TO_ONE[aa] for aa in true_aa]) if y_true is not None and true_aa else None,
                'accuracy': accuracy,
                'final_probabilities': final_probabilities.cpu().numpy()
            }

            # Add trajectory data if it was generated
            if args.detailed_json and 'trajectory_data' in locals():
                result_dict['trajectory_data'] = trajectory_data

            single_result = [result_dict]

            model_name = os.path.basename(args.model_path).replace('.pt', '')
            # Use protein name if provided, otherwise use structure index
            if args.protein_name:
                output_prefix = f"{args.output_prefix}_protein_{args.protein_name.replace('.', '_')}" if args.output_prefix else f"protein_{args.protein_name.replace('.', '_')}"
            else:
                output_prefix = f"{args.output_prefix}_single_struct_{actual_structure_idx}" if args.output_prefix else f"single_struct_{actual_structure_idx}"

            file_info = save_results_to_files(
                single_result, output_prefix, args.output_dir,
                model_name=model_name, split=args.split, steps=args.steps, T=args.T
            )

            # Generate detailed JSON output for single protein
            if args.detailed_json and 'trajectory_data' in locals():
                json_filepath = generate_detailed_json_output(
                    single_result, [structure_name], args.output_dir,
                    f"{args.output_prefix}_single_{structure_name.replace('.', '_')}" if args.output_prefix else f"single_{structure_name.replace('.', '_')}"
                )

            results = single_result
            structure_names = [structure_name]

    # Save comprehensive results
    print(f"\nSAMPLING SUMMARY:")
    print("="*60)
    print(f"Total structures: {len(results)}")
    successful_results = [r for r in results if 'error' not in r]
    print(f"Successful samples: {len(successful_results)}")
    print(f"Failed samples: {len(results) - len(successful_results)}")

    if successful_results:
        accuracies = [r.get('accuracy') for r in successful_results if r.get('accuracy') is not None]
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            print(f"Average accuracy: {avg_accuracy:.1f}%")

        print(f"\nFirst few examples:")
        for i, result in enumerate(successful_results[:3]):
            if 'predicted_sequence' in result:
                pred_seq = result['predicted_sequence']
                if isinstance(pred_seq, list):
                    # Convert indices to string
                    pred_seq_str = ''.join([THREE_TO_ONE[IDX_TO_AA[idx]] if 0 <= idx < len(IDX_TO_AA) else 'X' for idx in pred_seq])
                else:
                    pred_seq_str = str(pred_seq)

                acc_str = f"{result['accuracy']:.1f}%" if result.get('accuracy') is not None else "N/A"
                print(f"  Structure {i}: {pred_seq_str}")
                print(f"    Accuracy: {acc_str}")

    # Save files
    if len(results) > 1 or not args.detailed_json:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_prefix_final = args.output_prefix or "protein_sampling"

        file_info = save_results_to_files(
            results, output_prefix_final, args.output_dir,
            model_name=os.path.basename(args.model_path).replace('.pt', ''),
            split=args.split, steps=args.steps, T=args.T
        )

        print(f"\nFiles saved:")

        # Check if we have multiple sample files or single files
        if 'sample_files' in file_info:
            # Multiple samples per protein case
            print(f"Multiple samples detected ({len(file_info['sample_files'])} sample sets):")
            for sample_idx, sample_file_info in file_info['sample_files'].items():
                print(f"  Sample {sample_idx+1}:")
                print(f"    Sequences (CSV): {sample_file_info['sequences_file']}")
                print(f"    Probabilities (NPZ): {sample_file_info['probabilities_file']}")
                print(f"    Metadata (TXT): {sample_file_info['metadata_file']}")
        else:
            # Single file set case (backward compatibility)
            print(f"  Sequences (CSV): {file_info['sequences_file']}")
            print(f"  Probabilities (NPZ): {file_info['probabilities_file']}")
            print(f"  Metadata (TXT): {file_info['metadata_file']}")

        if args.detailed_json and len(results) > 1:
            json_filepath = generate_detailed_json_output(
                results, structure_names, args.output_dir, output_prefix_final
            )
            print(f"  Detailed JSON: {json_filepath}")

        print(f"  Timestamp: {file_info['timestamp']}")

    print("="*60)
    print("Sampling completed successfully!")


if __name__ == '__main__':
    main()
