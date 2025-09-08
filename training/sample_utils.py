def create_structural_ensemble(input_spec, ensemble_size=1, structure_noise_mag_std=1.0,
                              uncertainty_struct_noise_scaling=False, device='cpu', args=None, dataset_params=None):
    """
    Create an ensemble of structures from a single input by adding uncertainty-scaled noise.
    Uses GraphBuilder's built-in noise functionality.
    """
    from data.graph_builder import GraphBuilder
    from torch_geometric.data import Batch
    import torch
    import numpy as np
    
    # Input spec is already a processed protein dictionary when called from ensemble mode
    entry = input_spec
    temp_files = []
    
    try:
        # Parameter precedence: CLI args first, then model checkpoint as fallback
        graph_builder_kwargs = {}
        
        # Helper function to resolve parameter with proper precedence
        def resolve_param(param_name, cli_value, checkpoint_value, default_value=None):
            if cli_value is not None:
                return cli_value, "CLI args"
            elif checkpoint_value is not None:
                return checkpoint_value, "checkpoint"
            else:
                return default_value, "default"
        
        # Resolve each parameter with CLI precedence
        if args:
            k_val, k_source = resolve_param('k', getattr(args, 'k_neighbors', None), 
                                           dataset_params.get('k_neighbors') if dataset_params else None)
            k_farthest_val, k_farthest_source = resolve_param('k_farthest', getattr(args, 'k_farthest', None),
                                                             dataset_params.get('k_farthest') if dataset_params else None)
            k_random_val, k_random_source = resolve_param('k_random', getattr(args, 'k_random', None),
                                                         dataset_params.get('k_random') if dataset_params else None)
            max_edge_dist_val, max_edge_dist_source = resolve_param('max_edge_dist', getattr(args, 'max_edge_dist', None),
                                                                   dataset_params.get('max_edge_dist') if dataset_params else None)
            rbf_3d_min_val, rbf_3d_min_source = resolve_param('rbf_3d_min', getattr(args, 'rbf_3d_min', None),
                                                             dataset_params.get('rbf_3d_min') if dataset_params else None, 2.0)
            rbf_3d_max_val, rbf_3d_max_source = resolve_param('rbf_3d_max', getattr(args, 'rbf_3d_max', None),
                                                             dataset_params.get('rbf_3d_max') if dataset_params else None, 350.0)
            rbf_3d_spacing_val, rbf_3d_spacing_source = resolve_param('rbf_3d_spacing', getattr(args, 'rbf_3d_spacing', None),
                                                                     dataset_params.get('rbf_3d_spacing') if dataset_params else None, 'exponential')
            
            graph_builder_kwargs = {
                'k': k_val,
                'k_farthest': k_farthest_val,
                'k_random': k_random_val,
                'max_edge_dist': max_edge_dist_val,
                'num_rbf_3d': dataset_params.get('num_rbf_3d', 16) if dataset_params else 16,
                'num_rbf_seq': dataset_params.get('num_rbf_seq', 16) if dataset_params else 16,
                'use_virtual_node': dataset_params.get('use_virtual_node', True) if dataset_params else True,
                'no_source_indicator': dataset_params.get('no_source_indicator', False) if dataset_params else False,
                'rbf_3d_min': rbf_3d_min_val,
                'rbf_3d_max': rbf_3d_max_val,
                'rbf_3d_spacing': rbf_3d_spacing_val,
            }
            
            if getattr(args, 'verbose', False):
                print(f"[DEBUG] GraphBuilder parameter sources:")
                print(f"  k: {k_val} (source: {k_source})")
                print(f"  k_farthest: {k_farthest_val} (source: {k_farthest_source})")
                print(f"  k_random: {k_random_val} (source: {k_random_source})")
                print(f"  max_edge_dist: {max_edge_dist_val} (source: {max_edge_dist_source})")
                print(f"  rbf_3d_min: {rbf_3d_min_val} (source: {rbf_3d_min_source})")
                print(f"  rbf_3d_max: {rbf_3d_max_val} (source: {rbf_3d_max_source})")
        else:
            # Fallback to checkpoint only if no args
            if dataset_params:
                graph_builder_kwargs = {
                    'k': dataset_params.get('k_neighbors'),
                    'k_farthest': dataset_params.get('k_farthest'),
                    'k_random': dataset_params.get('k_random'),
                    'max_edge_dist': dataset_params.get('max_edge_dist'),
                    'num_rbf_3d': dataset_params.get('num_rbf_3d', 16),
                    'num_rbf_seq': dataset_params.get('num_rbf_seq', 16),
                    'use_virtual_node': dataset_params.get('use_virtual_node', True),
                    'no_source_indicator': dataset_params.get('no_source_indicator', False),
                    'rbf_3d_min': dataset_params.get('rbf_3d_min', 2.0),
                    'rbf_3d_max': dataset_params.get('rbf_3d_max', 350.0),
                    'rbf_3d_spacing': dataset_params.get('rbf_3d_spacing', 'exponential'),
                }
        
        # Add noise settings to graph builder (these are ensemble-specific)
        graph_builder_kwargs['structure_noise_mag_std'] = structure_noise_mag_std
        graph_builder_kwargs['uncertainty_struct_noise_scaling'] = uncertainty_struct_noise_scaling
        graph_builder_kwargs['time_based_struct_noise'] = 'fixed'  # No time-based scaling for ensemble
        
        print(f"Creating ensemble of {ensemble_size} replicas")
        if structure_noise_mag_std > 0:
            print(f"  Noise std: {structure_noise_mag_std} Å")
            if uncertainty_struct_noise_scaling:
                print(f"  Uncertainty-scaled noise enabled")
        
        if args and getattr(args, 'verbose', False):
            print(f"[DEBUG] Full GraphBuilder kwargs for ensemble:")
            for key, value in graph_builder_kwargs.items():
                print(f"  {key}: {value}")
        
        # Create graph builder with noise settings
        graph_builder = GraphBuilder(**graph_builder_kwargs)
        
        # Generate ensemble members
        ensemble_graphs = []
        previous_coords = None
        
        for replica_idx in range(ensemble_size):
            # Create a fresh GraphBuilder for each replica to prevent state pollution
            replica_graph_builder = GraphBuilder(**graph_builder_kwargs)
            
            # Build graph (GraphBuilder will apply noise internally)
            graph = replica_graph_builder.build_from_dict(entry.copy(), time_param=0.0)
            graph.ensemble_idx = replica_idx
            
            if args and getattr(args, 'verbose', False) and replica_idx == 0:
                print(f"[DEBUG] Replica {replica_idx} graph structure:")
                print(f"  num_nodes: {graph.num_nodes}")
                print(f"  num_edges: {graph.num_edges}")
                print(f"  pos.shape: {graph.pos.shape}")
                if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                    print(f"  edge_attr.shape: {graph.edge_attr.shape}")
                else:
                    print(f"  edge_attr: None")
            
            # Verify diversity - check that noise was actually different
            if structure_noise_mag_std > 0 and replica_idx > 0:
                # Compare positions with first replica
                if replica_idx == 1:
                    # Store first replica's coordinates for comparison
                    previous_coords = ensemble_graphs[0].pos.clone()
                
                current_coords = graph.pos
                coord_diff = torch.norm(current_coords - previous_coords).item()
                
                if coord_diff < 1e-6:
                    raise RuntimeError(
                        f"CRITICAL ERROR: Replica {replica_idx} has identical coordinates to replica 0!\n"
                        f"Coordinate difference: {coord_diff:.2e} Å\n"
                        f"This indicates a random seed issue - all replicas are getting the same noise.\n"
                        f"Please check PyTorch random seed settings or file a bug report."
                    )
                
                if args and getattr(args, 'verbose', False):
                    print(f"  Replica {replica_idx} coordinate RMSD from replica 0: {coord_diff:.3f} Å")
            
            ensemble_graphs.append(graph)
        
        # Batch all graphs together
        batched_ensemble = Batch.from_data_list(ensemble_graphs)
        batched_ensemble.ensemble_size = ensemble_size
        
        return batched_ensemble.to(device)
        
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass


def sample_with_ensemble_consensus(model, batched_ensemble, T=8.0, t_min=0.0, steps=50, K=21,
                                  consensus_strength=0.2, device='cpu', use_virtual_node=True, args=None):
    """
    Sample from an ensemble with state aggregation at each timestep.
    
    Key design: We aggregate STATES (current probabilities) not PREDICTIONS.
    This maintains diversity while reducing overconfidence.
    
    Args:
        model: Trained model
        batched_ensemble: Batched graph from create_structural_ensemble
        T: Maximum time
        t_min: Minimum time
        steps: Number of sampling steps
        K: Number of amino acid classes
        consensus_strength: How much to blend states (0=independent, 1=full consensus)
        device: PyTorch device
        args: Additional arguments
        
    Returns:
        Final consensus sequence as list of amino acid indices
    """
    from torch.distributions import Dirichlet
    from torch_geometric.data import Batch
    import torch
    from tqdm import tqdm
    
    model.eval()
    model = model.to(device)
    batched_ensemble = batched_ensemble.to(device)
    
    ensemble_size = batched_ensemble.ensemble_size
    
    # Get sequence length (should be same for all replicas)
    seq_lengths = []
    for i in range(ensemble_size):
        mask = (batched_ensemble.batch == i)
        num_nodes = mask.sum().item()
        # Handle virtual nodes - use the parameter passed to the function
        seq_len = num_nodes - 1 if use_virtual_node else num_nodes
        seq_lengths.append(seq_len)
    
    # Verify all replicas have same length
    if len(set(seq_lengths)) != 1:
        raise ValueError(f"Ensemble members have different lengths: {seq_lengths}")
    
    N = seq_lengths[0]
    
    print(f"\nSampling with {ensemble_size}-member ensemble, consensus_strength={consensus_strength}")
    
    # Initialize with different Dirichlet noise for each replica
    dirichlet_concentration = getattr(args, 'dirichlet_concentration', 20.0) if args else 20.0
    dirichlet_dist = Dirichlet(dirichlet_concentration * torch.ones(K, device=device))
    
    # Sample different initial states for each replica
    x_batch = dirichlet_dist.sample((ensemble_size, N))  # [ensemble_size, N, K]
    
    if args and getattr(args, 'verbose', False):
        print(f"[DEBUG] Initial Dirichlet states for ensemble:")
        print(f"  x_batch.shape: {x_batch.shape}")
        print(f"  dirichlet_concentration: {dirichlet_concentration}")
        for i in range(min(2, ensemble_size)):
            replica_entropy = -(x_batch[i] * torch.log(x_batch[i] + 1e-8)).sum(-1).mean()
            print(f"  Replica {i} initial entropy: {replica_entropy:.4f}")
            print(f"  Replica {i} first 3 pos max probs: {[x_batch[i, j].max().item() for j in range(3)]}")
        
        # Check if replicas are actually different
        if ensemble_size > 1:
            state_diff = torch.norm(x_batch[0] - x_batch[1]).item()
            print(f"  L2 difference between replica 0 and 1 initial states: {state_diff:.6f}")
            if state_diff < 1e-6:
                print(f"  WARNING: Initial states are nearly identical!")
            else:
                print(f"  ✓ Initial states are properly diversified")
    
    # Time integration
    times = torch.linspace(t_min, T, steps, device=device)
    dt = (T - t_min) / (steps - 1) if steps > 1 else 0
    
    with torch.no_grad():
        for step_idx, t_val in enumerate(tqdm(times[:-1], desc=f"Ensemble sampling")):
            
            # State consensus (aggregate current states before prediction)
            if consensus_strength > 0:
                # Get ensemble method from args, default to arithmetic
                ensemble_method = getattr(args, 'ensemble_method', 'arithmetic') if args else 'arithmetic'
                
                # Apply consensus blending using the specified method
                x_batch = compute_ensemble_consensus(x_batch, consensus_strength, ensemble_method, ensemble_size)
            
            # Get predictions from model using batched processing
            t_tensor = torch.full((ensemble_size,), t_val, device=device)
            
            if args and getattr(args, 'verbose', False) and step_idx == 0:
                print(f"[DEBUG] Computing predictions with BATCHED processing:")
                print(f"  batched_ensemble.num_graphs: {batched_ensemble.num_graphs}")
                print(f"  batched_ensemble.num_nodes: {batched_ensemble.num_nodes}")
                print(f"  x_batch.shape: {x_batch.shape}")
                print(f"  t_tensor.shape: {t_tensor.shape}")
            
            # Single batched forward pass for all replicas at once!
            model_output = model(batched_ensemble, t_tensor, x_batch)
            
            # Handle DSSP multitask output
            if isinstance(model_output, tuple):
                logits = model_output[0]  # [total_nodes, K]
            else:
                logits = model_output  # [total_nodes, K]
            
            # Apply temperature
            flow_temp = getattr(args, 'flow_temp', 1.0) if args else 1.0
            pred_target = torch.softmax(logits / flow_temp, dim=-1)  # [total_nodes, K]
            
            if args and getattr(args, 'verbose', False) and step_idx == 0:
                print(f"[DEBUG] Batched predictions:")
                print(f"  logits.shape: {logits.shape}")
                print(f"  pred_target.shape: {pred_target.shape}")
                print(f"  logits stats: min={logits.min():.3f}, max={logits.max():.3f}, mean={logits.mean():.3f}")
                print(f"  pred_target stats: min={pred_target.min():.6f}, max={pred_target.max():.6f}")
            
            # Split predictions back into per-replica format [ensemble_size, N, K]
            pred_ensemble = torch.zeros((ensemble_size, N, K), dtype=torch.float32, device=device)
            
            # Extract predictions for each replica from batched output
            node_offset = 0
            for replica_idx in range(ensemble_size):
                # Each replica has N+1 nodes (N real + 1 virtual) if use_virtual_node
                nodes_per_replica = N + 1 if use_virtual_node else N
                
                # Extract this replica's predictions
                replica_predictions = pred_target[node_offset:node_offset + nodes_per_replica]  # [N+1, K] or [N, K]
                
                # Handle virtual nodes
                if use_virtual_node:
                    pred_real = replica_predictions[:N]  # Take only real nodes [N, K]
                else:
                    pred_real = replica_predictions  # [N, K]
                
                pred_ensemble[replica_idx] = pred_real
                node_offset += nodes_per_replica
                
                if args and getattr(args, 'verbose', False) and step_idx == 0 and replica_idx == 0:
                    print(f"[DEBUG] Replica {replica_idx} extracted predictions:")
                    print(f"  pred_real.shape: {pred_real.shape}")
                    print(f"  pred_real first 5 positions max probs: {[pred_real[i].max().item() for i in range(min(5, pred_real.shape[0]))]}")
            
            # Apply consensus if needed
            if consensus_strength > 0:
                # Apply consensus averaging
                consensus_weights = torch.softmax(consensus_strength * torch.ones(ensemble_size, device=device), dim=0)
                pred_probs = torch.sum(consensus_weights.view(-1, 1, 1) * pred_ensemble, dim=0, keepdim=True)
                pred_probs = pred_probs.expand(ensemble_size, -1, -1)  # Broadcast to all replicas
            else:
                # No consensus - use individual predictions
                pred_probs = pred_ensemble
            
            # Compute velocities using batched processing
            use_smoothed_targets = getattr(args, 'use_smoothed_targets', False) if args else False
            use_c_factor = getattr(args, 'use_c_factor', False) if args else False
            
            if args and getattr(args, 'verbose', False) and step_idx == 0:
                print(f"[DEBUG] Computing velocities with batched processing:")
                print(f"  x_batch.shape: {x_batch.shape}")
                print(f"  pred_probs.shape: {pred_probs.shape}")
                print(f"  t_tensor.shape: {t_tensor.shape}")
            
            # Compute velocities for all replicas at once (much faster!)
            velocities = model.cond_flow.velocity(
                x_batch, pred_probs, t_tensor,
                use_virtual_node=use_virtual_node,
                use_smoothed_targets=use_smoothed_targets,
                use_c_factor=use_c_factor
            )
            
            if args and getattr(args, 'verbose', False) and step_idx == 0:
                print(f"[DEBUG] Batched velocities:")
                print(f"  velocities.shape: {velocities.shape}")
                print(f"  velocities stats: min={velocities.min():.6f}, max={velocities.max():.6f}, mean={velocities.mean():.6f}")
            
            # Update states with Euler step
            x_new = x_batch + dt * velocities
            x_batch = simplex_proj(x_new)
            
            # Debug: Check if replicas remain different after update
            if args and getattr(args, 'verbose', False) and ensemble_size > 1:
                state_diff = torch.norm(x_batch[0] - x_batch[1]).item()
                print(f"[DEBUG] After step {step_idx}: L2 diff between replicas = {state_diff:.6f}")
                if state_diff < 1e-6:
                    print(f"[DEBUG] WARNING: Replicas have become identical after update!")
                    print(f"[DEBUG] Replica 0 first 5 max probs: {[x_batch[0, i].max().item() for i in range(5)]}")
                    print(f"[DEBUG] Replica 1 first 5 max probs: {[x_batch[1, i].max().item() for i in range(5)]}")
                elif state_diff < 0.1:
                    print(f"[DEBUG] WARNING: Replicas are converging!")
                else:
                    print(f"[DEBUG] Replicas remain different")
    
    # Final sequence generation based on consensus strength
    if args and getattr(args, 'verbose', False):
        print(f"[DEBUG] Final consensus decision: consensus_strength={consensus_strength}")
    
    if consensus_strength > 0:
        # Consensus: average all replica probabilities using specified method
        ensemble_method = getattr(args, 'ensemble_method', 'arithmetic') if args else 'arithmetic'
        
        if args and getattr(args, 'verbose', False):
            print(f"[DEBUG] Applying final consensus averaging using {ensemble_method} method")
        
        # Apply the chosen consensus method directly for final sequence generation
        if ensemble_method == 'arithmetic':
            # Standard arithmetic mean in probability space
            final_consensus = x_batch.mean(dim=0)  # [N, K]
        elif ensemble_method == 'geometric':
            # Geometric mean in log space
            eps = 1e-8
            log_states = torch.log(x_batch + eps)  # [ensemble_size, N, K]
            log_mean = log_states.mean(dim=0)  # [N, K]
            final_consensus = torch.exp(log_mean)  # [N, K]
            # Normalize to ensure it's on the simplex
            final_consensus = final_consensus / final_consensus.sum(dim=-1, keepdim=True)
        else:
            # Fallback to arithmetic if unknown method
            final_consensus = x_batch.mean(dim=0)  # [N, K]
            
        final_sequence = final_consensus.argmax(-1).tolist()
    else:
        # No consensus: use first replica only (but let's debug all replicas)
        if args and getattr(args, 'verbose', False):
            print(f"[DEBUG] Using first replica only (no consensus)")
            print(f"[DEBUG] Debugging all replica sequences (first 20 positions):")
            for replica_idx in range(min(ensemble_size, 3)):  # Show up to 3 replicas
                replica_seq = x_batch[replica_idx].argmax(-1).tolist()
                print(f"  Replica {replica_idx}: {replica_seq[:20]}")
        
        final_sequence = x_batch[0].argmax(-1).tolist()
    
    return final_sequence


def adjust_batch_size_for_ensemble(batch_size, ensemble_size):
    """
    Adjust batch size to avoid splitting proteins across batches in ensemble mode.
    
    Args:
        batch_size: Desired batch size
        ensemble_size: Number of replicas per protein
        
    Returns:
        Adjusted batch size that's divisible by ensemble_size
    """
    if batch_size % ensemble_size == 0:
        return batch_size
    
    # Find the largest batch size <= original that's divisible by ensemble_size
    adjusted_batch_size = (batch_size // ensemble_size) * ensemble_size
    
    if adjusted_batch_size == 0:
        adjusted_batch_size = ensemble_size
    
    print(f"Adjusted batch size from {batch_size} to {adjusted_batch_size} to avoid splitting proteins across batches (ensemble_size={ensemble_size})")
    
    return adjusted_batch_size


def sample_multiple_proteins_with_ensemble(model, dataset, indices=None, steps=50, T=8.0, t_min=0.0, K=21,
                                         ensemble_size=5, consensus_strength=0.2,
                                         structure_noise_mag_std=0.0, uncertainty_struct_noise_scaling=False,
                                         batch_size=32, args=None):
    """
    Sample multiple proteins with ensemble consensus using mega-batching strategy.

    
    Args:
        model: Trained DFM model
        dataset: CathDataset instance
        indices: List of structure indices to sample (None = all)
        steps: Number of sampling steps
        T: Maximum time
        t_min: Minimum time
        K: Number of amino acid classes
        ensemble_size: Number of replicas per protein
        consensus_strength: How much to blend states (0=independent, 1=full consensus)
        structure_noise_mag_std: Standard deviation for structural noise
        uncertainty_struct_noise_scaling: Whether to scale noise by uncertainty
        batch_size: Desired batch size (will be adjusted to be divisible by ensemble_size)
        args: Arguments object
        
    Returns:
        List of result dictionaries with evaluation metrics
    """
    from torch.distributions import Dirichlet
    from torch_geometric.data import Batch
    from tqdm import tqdm
    
    if indices is None:
        indices = list(range(len(dataset)))
    
    # Adjust batch size to avoid splitting proteins across batches
    adjusted_batch_size = adjust_batch_size_for_ensemble(batch_size, ensemble_size)
    proteins_per_batch = adjusted_batch_size // ensemble_size
    
    print(f"\nEnsemble Multi-Protein Sampling Configuration:")
    print(f"  Total proteins: {len(indices)}")
    print(f"  Ensemble size: {ensemble_size}")
    print(f"  Consensus strength: {consensus_strength}")
    print(f"  Adjusted batch size: {adjusted_batch_size} (from {batch_size})")
    print(f"  Proteins per batch: {proteins_per_batch}")
    print(f"  Graphs per batch: {adjusted_batch_size} ({proteins_per_batch} proteins × {ensemble_size} replicas)")
    print(f"  Structure noise std: {structure_noise_mag_std} Å")
    
    # Calculate number of batches needed
    num_batches = (len(indices) + proteins_per_batch - 1) // proteins_per_batch
    print(f"  Total batches needed: {num_batches}")
    
    # Show batch breakdown for verification
    if getattr(args, 'verbose', False):
        print(f"  Batch breakdown:")
        for batch_idx in range(num_batches):
            batch_start = batch_idx * proteins_per_batch
            batch_end = min(batch_start + proteins_per_batch, len(indices))
            proteins_in_this_batch = batch_end - batch_start
            graphs_in_this_batch = proteins_in_this_batch * ensemble_size
            print(f"    Batch {batch_idx + 1}: {proteins_in_this_batch} proteins, {graphs_in_this_batch} graphs")
    
    model.eval()
    device = next(model.parameters()).device
    results = []
    
    # Extract dataset parameters for ensemble creation
    dataset_params = None
    if hasattr(dataset, 'graph_builder'):
        graph_builder = dataset.graph_builder
        dataset_params = {
            'k_neighbors': getattr(graph_builder, 'k', None),
            'k_farthest': getattr(graph_builder, 'k_farthest', None),
            'k_random': getattr(graph_builder, 'k_random', None),
            'max_edge_dist': getattr(graph_builder, 'max_edge_dist', None),
            'num_rbf_3d': getattr(graph_builder, 'num_rbf_3d', 16),
            'num_rbf_seq': getattr(graph_builder, 'num_rbf_seq', 16),
            'use_virtual_node': getattr(graph_builder, 'use_virtual_node', True),
            'no_source_indicator': getattr(graph_builder, 'no_source_indicator', False),
            'rbf_3d_min': getattr(graph_builder, 'rbf_3d_min', 2.0),
            'rbf_3d_max': getattr(graph_builder, 'rbf_3d_max', 350.0),
            'rbf_3d_spacing': getattr(graph_builder, 'rbf_3d_spacing', 'exponential'),
        }
    
    # Batch processing loop (only when total_proteins > proteins_per_batch)
    for batch_idx in range(num_batches):
        batch_start = batch_idx * proteins_per_batch
        batch_end = min(batch_start + proteins_per_batch, len(indices))
        batch_indices = indices[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}: proteins {batch_start}-{batch_end-1}")
        
        try:
            # Create mega-batch with all protein ensembles
            print(f"  Creating ensemble mega-batch...")
            
            mega_batch_graphs = []
            protein_metadata = []
            
            for protein_pos, dataset_idx in enumerate(batch_indices):
                try:
                    # Get protein data from dataset
                    data, y_true, mask, time_value, dssp_targets = dataset[dataset_idx]
                    structure_name = getattr(data, 'name', f'structure_{dataset_idx}')
                    
                    # Get the original entry from dataset for ensemble creation
                    if hasattr(dataset, 'entries'):
                        # CathDataset has entries list
                        original_entry = dataset.entries[dataset_idx]
                    elif hasattr(dataset, 'protein_entry'):
                        # SingleProteinDataset has single protein_entry
                        original_entry = dataset.protein_entry
                    else:
                        raise AttributeError(f"Cannot access original entry from dataset type: {type(dataset)}")
                    
                    # Create ensemble for this protein using original entry
                    ensemble_batch = create_structural_ensemble(
                        original_entry,
                        ensemble_size=ensemble_size,
                        structure_noise_mag_std=structure_noise_mag_std,
                        uncertainty_struct_noise_scaling=uncertainty_struct_noise_scaling,
                        device=device,
                        args=args,
                        dataset_params=dataset_params
                    )
                    
                    # Add ensemble graphs to mega batch
                    ensemble_graphs = ensemble_batch.to_data_list()
                    for replica_idx, graph in enumerate(ensemble_graphs):
                        graph.protein_idx = dataset_idx
                        graph.replica_idx = replica_idx
                        graph.protein_pos = protein_pos  # Position within this batch
                        mega_batch_graphs.append(graph)
                    
                    # Get sequence length from original entry
                    seq_len = len(original_entry['seq'])
                    
                    # Store metadata
                    protein_metadata.append({
                        'dataset_idx': dataset_idx,
                        'structure_name': structure_name,
                        'y_true': y_true,
                        'seq_len': seq_len,
                        'protein_pos': protein_pos
                    })
                
                except Exception as e:
                    print(f"    Error creating ensemble for protein {dataset_idx}: {e}")
                    # Add error result and continue
                    results.append({
                        'structure_idx': dataset_idx,
                        'structure_name': f'structure_{dataset_idx}',
                        'error': str(e),
                        'ensemble_size': ensemble_size
                    })
                    continue
            
            if not mega_batch_graphs:
                print(f"    No valid graphs in batch, skipping...")
                continue
            
            # Create mega-batch
            print(f"    Batching {len(mega_batch_graphs)} graphs ({len(protein_metadata)} proteins × {ensemble_size} replicas)...")
            mega_batch = Batch.from_data_list(mega_batch_graphs).to(device)
            
            # Initialize states for all proteins × replicas
            use_virtual_node = dataset_params.get('use_virtual_node', True) if dataset_params else True
            dirichlet_concentration = getattr(args, 'dirichlet_concentration', 20.0) if args else 20.0
            flow_temp = getattr(args, 'flow_temp', 1.0) if args else 1.0
            
            # Get max sequence length in this batch
            max_seq_len = max(meta['seq_len'] for meta in protein_metadata)
            total_graphs = len(mega_batch_graphs)
            
            # Initialize Dirichlet states: [total_graphs, max_seq_len, K]
            dirichlet_dist = Dirichlet(dirichlet_concentration * torch.ones(K, device=device))
            x_mega = torch.zeros(total_graphs, max_seq_len, K, device=device)
            
            graph_idx = 0
            for protein_meta in protein_metadata:
                seq_len = protein_meta['seq_len']
                for replica_idx in range(ensemble_size):
                    # Sample different initial state for each replica
                    x_init = dirichlet_dist.sample((seq_len,))  # [seq_len, K]
                    x_mega[graph_idx, :seq_len, :] = x_init
                    graph_idx += 1
            
            print(f"    Initialized {total_graphs} graph states with max_seq_len={max_seq_len}")
            
            # Time integration loop
            times = torch.linspace(t_min, T, steps, device=device)
            dt = (T - t_min) / (steps - 1) if steps > 1 else 0
            
            with torch.no_grad():
                for step_idx, t_val in enumerate(tqdm(times[:-1], desc=f"Batch {batch_idx + 1} ensemble sampling")):
                    
                    # Apply consensus within each protein's ensemble (vectorized)
                    if consensus_strength > 0:
                        # Get ensemble method from args, default to arithmetic
                        ensemble_method = getattr(args, 'ensemble_method', 'arithmetic') if args else 'arithmetic'
                        
                        protein_graph_idx = 0
                        for protein_meta in protein_metadata:
                            seq_len = protein_meta['seq_len']
                            
                            # Extract this protein's replicas
                            protein_states = x_mega[protein_graph_idx:protein_graph_idx + ensemble_size, :seq_len, :]  # [ensemble_size, seq_len, K]
                            
                            # Apply consensus blending using the specified method
                            blended_states = compute_ensemble_consensus(protein_states, consensus_strength, ensemble_method, ensemble_size)
                            
                            # Write back to mega tensor
                            x_mega[protein_graph_idx:protein_graph_idx + ensemble_size, :seq_len, :] = blended_states
                            protein_graph_idx += ensemble_size
                    
                    # Mega-batched model prediction (single forward pass for ALL graphs!)
                    t_tensor = torch.full((total_graphs,), t_val, device=device)
                    model_output = model(mega_batch, t_tensor, x_mega)
                    
                    # Handle DSSP multitask output
                    if isinstance(model_output, tuple):
                        logits = model_output[0]
                    else:
                        logits = model_output
                    
                    pred_target = torch.softmax(logits / flow_temp, dim=-1)
                    
                    # Split predictions back to [total_graphs, max_seq_len, K] format
                    pred_mega = torch.zeros(total_graphs, max_seq_len, K, device=device)
                    
                    node_offset = 0
                    for graph_idx in range(total_graphs):
                        # Count nodes for this graph
                        mask = (mega_batch.batch == graph_idx)
                        num_nodes = mask.sum().item()
                        
                        # Extract predictions for this graph
                        graph_predictions = pred_target[node_offset:node_offset + num_nodes]
                        
                        # Handle virtual nodes
                        if use_virtual_node and num_nodes > 0:
                            real_nodes = num_nodes - 1
                            seq_len = min(real_nodes, max_seq_len)
                            pred_mega[graph_idx, :seq_len, :] = graph_predictions[:seq_len]
                        else:
                            seq_len = min(num_nodes, max_seq_len)
                            pred_mega[graph_idx, :seq_len, :] = graph_predictions[:seq_len]
                        
                        node_offset += num_nodes
                    
                    # Mega-batched velocity computation (single call for ALL graphs!)
                    velocities = model.cond_flow.velocity(
                        x_mega, pred_mega, t_tensor,
                        use_virtual_node=use_virtual_node,
                        use_smoothed_targets=getattr(args, 'use_smoothed_targets', False) if args else False,
                        use_c_factor=getattr(args, 'use_c_factor', False) if args else False
                    )
                    
                    # Update states with Euler step
                    x_new = x_mega + dt * velocities
                    x_mega = simplex_proj(x_new)
            
            # Extract results per protein
            print(f"    Generating final sequences...")
            
            protein_graph_idx = 0
            for protein_meta in protein_metadata:
                try:
                    seq_len = protein_meta['seq_len']
                    
                    # Extract this protein's replica states
                    protein_states = x_mega[protein_graph_idx:protein_graph_idx + ensemble_size, :seq_len, :]  # [ensemble_size, seq_len, K]
                    
                    # Generate consensus sequence for this protein
                    if consensus_strength > 0:
                        # Get ensemble method from args, default to arithmetic
                        ensemble_method = getattr(args, 'ensemble_method', 'arithmetic') if args else 'arithmetic'
                        
                        # Apply the chosen consensus method directly for final sequence generation
                        if ensemble_method == 'arithmetic':
                            # Standard arithmetic mean in probability space
                            final_consensus = protein_states.mean(dim=0)  # [seq_len, K]
                        elif ensemble_method == 'geometric':
                            # Geometric mean in log space
                            eps = 1e-8
                            log_states = torch.log(protein_states + eps)  # [ensemble_size, seq_len, K]
                            log_mean = log_states.mean(dim=0)  # [seq_len, K]
                            final_consensus = torch.exp(log_mean)  # [seq_len, K]
                            # Normalize to ensure it's on the simplex
                            final_consensus = final_consensus / final_consensus.sum(dim=-1, keepdim=True)
                        else:
                            # Fallback to arithmetic if unknown method
                            final_consensus = protein_states.mean(dim=0)  # [seq_len, K]
                            
                        final_sequence = final_consensus.argmax(-1).tolist()
                    else:
                        final_sequence = protein_states[0].argmax(-1).tolist()  # Use first replica
                    
                    # Calculate accuracy if ground truth available
                    accuracy = None
                    true_seq = None
                    if protein_meta['y_true'] is not None:
                        y_true = protein_meta['y_true']
                        if use_virtual_node and y_true.shape[0] > seq_len:
                            y_true = y_true[:seq_len]  # Remove virtual node
                        
                        true_seq = y_true.argmax(-1).tolist()
                        correct = sum(p == t for p, t in zip(final_sequence, true_seq))
                        accuracy = correct / len(final_sequence) * 100
                    
                    # Convert to amino acid names
                    predicted_aa = []
                    for idx_val in final_sequence:
                        if 0 <= idx_val < len(IDX_TO_AA):
                            predicted_aa.append(IDX_TO_AA[idx_val])
                        else:
                            predicted_aa.append('XXX')
                    
                    result = {
                        'structure_idx': protein_meta['dataset_idx'],
                        'structure_name': protein_meta['structure_name'],
                        'ensemble_size': ensemble_size,
                        'consensus_strength': consensus_strength,
                        'length': len(final_sequence),
                        'predicted_indices': final_sequence,
                        'predicted_aa': predicted_aa,
                        'predicted_sequence': ''.join([THREE_TO_ONE.get(aa, 'X') for aa in predicted_aa]),
                        'true_indices': true_seq,
                        'accuracy': accuracy,
                        'final_probabilities': (final_consensus.cpu().numpy() if consensus_strength > 0 
                                              else protein_states[0].cpu().numpy())
                    }
                    
                    results.append(result)
                    protein_graph_idx += ensemble_size
                    
                except Exception as e:
                    print(f"      Error generating sequence for protein {protein_meta['dataset_idx']}: {e}")
                    results.append({
                        'structure_idx': protein_meta['dataset_idx'],
                        'structure_name': protein_meta['structure_name'],
                        'error': str(e),
                        'ensemble_size': ensemble_size
                    })
                    protein_graph_idx += ensemble_size
        
        except Exception as e:
            print(f"    Error processing batch {batch_idx}: {e}")
            # Add error results for all proteins in this batch
            for idx in batch_indices:
                results.append({
                    'structure_idx': idx,
                    'structure_name': f'structure_{idx}',
                    'error': str(e),
                    'ensemble_size': ensemble_size
                })
    
    print(f"\nCompleted ensemble sampling: {len(results)} proteins with ensemble_size={ensemble_size}")
    return results


"""
Utility functions for protein sequence sampling.

This module contains helper functions, classes, and utilities that support
the core sampling functionality in sample.py.
"""

import os
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch.utils.data import Dataset
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm


# Amino acid mappings (constants used across sampling utilities)
THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'XXX': 'X'
}

AA_TO_IDX = {
    'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4, 'GLY': 5, 'HIS': 6, 'ILE': 7,
    'LYS': 8, 'LEU': 9, 'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14, 'SER': 15,
    'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19, 'XXX': 20
}

IDX_TO_AA = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE',
             'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER',
             'THR', 'VAL', 'TRP', 'TYR', 'XXX']

SINGLE_TO_TRIPLE = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
    'X': 'XXX'
}


# Input processing functions for PDB download and chain selection
def download_pdb(pdb_id: str, save_path: str = None) -> str:
    """Download PDB file from RCSB PDB database."""
    import requests
    import tempfile
    
    if save_path is None:
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, f"{pdb_id.lower()}.pdb")
    
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
    print(f"Downloading {pdb_id} from RCSB PDB...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'w') as f:
            f.write(response.text)
            
        print(f"Downloaded to: {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download PDB {pdb_id}: {e}")


def extract_chain_from_pdb(pdb_path: str, chain_id: str) -> str:
    """Extract specific chain from PDB file using BioPython."""
    try:
        from Bio import PDB
        output_path = f"{os.path.splitext(pdb_path)[0]}_chain{chain_id.upper()}.pdb"
        
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        
        # Find the specified chain
        target_chain = None
        for model in structure:
            if chain_id.upper() in [chain.id for chain in model]:
                target_chain = model[chain_id.upper()]
                break
        
        if target_chain is None:
            available_chains = []
            for model in structure:
                available_chains.extend([chain.id for chain in model])
            raise ValueError(f"Chain {chain_id} not found. Available chains: {list(set(available_chains))}")
        
        # Save only the specified chain
        io = PDB.PDBIO()
        io.set_structure(target_chain)
        io.save(output_path)
        
        print(f"Extracted chain {chain_id} to: {output_path}")
        return output_path
    except ImportError:
        raise ImportError("BioPython is required for chain extraction. Install with: pip install biopython")
    except Exception as e:
        raise RuntimeError(f"Failed to extract chain {chain_id}: {e}")


def pdb_to_dict_format(pdb_path: str, structure_name: str) -> dict:
    """
    Convert PDB file to dictionary format compatible with GraphBuilder.build_from_dict().
    Uses existing PDBProcessor to extract coordinates and B-factors.
    """
    from pathlib import Path
    from data.pdb_processor import PDBProcessor
    
    pdb_dir = os.path.dirname(pdb_path)
    processor = PDBProcessor(pdb_directory=pdb_dir, verbose=False)
    
    # Extract sequence and coordinates using existing method
    sequence, coords_dict = processor._extract_sequence_and_coords_from_file(Path(pdb_path))
    if sequence is None or coords_dict is None:
        raise ValueError(f"Failed to extract sequence/coordinates from {pdb_path}")
    
    # Extract B-factors using existing method  
    b_factors = processor._extract_b_factors_from_file(Path(pdb_path))
    if b_factors is None:
        # Use default B-factors if extraction fails
        b_factors = [50.0] * len(sequence)
        print(f"Warning: Using default B-factors for {structure_name}")
    
    # Convert to the format expected by GraphBuilder.build_from_dict()
    entry = {
        'name': structure_name,
        'seq': sequence,
        'coords': {
            'N': coords_dict['N'],
            'CA': coords_dict['CA'], 
            'C': coords_dict['C'],
            'O': coords_dict['O']
        },
        'b_factors': np.array(b_factors),
        'source': 'pdb'
    }
    
    return entry


def cif_to_dict_format(cif_path: str, structure_name: str) -> dict:
    """Convert CIF file to dictionary format using existing CIF parser."""
    from data.cif_parser import parse_cif_backbone_auto
    
    coords, scores, residue_types, source = parse_cif_backbone_auto(cif_path)
    
    # Convert to dict format
    coords_dict = {
        'N': coords[:, 0, :].numpy(),
        'CA': coords[:, 1, :].numpy(), 
        'C': coords[:, 2, :].numpy(),
        'O': coords[:, 3, :].numpy()
    }
    
    sequence = ''.join([THREE_TO_ONE.get(aa, 'X') for aa in residue_types])
    
    entry = {
        'name': structure_name,
        'seq': sequence,
        'coords': coords_dict,
        'b_factors': scores.numpy(),
        'source': source
    }
    
    return entry


def process_input_specification(input_spec: str, verbose=False) -> tuple:
    """
    Process input specification and return (structure_dict, temp_files_to_cleanup).
    
    Supported formats:
    - Local PDB files: '/path/to/file.pdb' or 'protein.pdb'
    - Local CIF files: '/path/to/file.cif' or 'protein.cif' 
    - PDB IDs: '1abc'
    - PDB ID with chain: '1abc.C'
    
    Returns:
        tuple: (structure_dict, temp_files_list)
    """
    import tempfile
    
    temp_files = []
    input_type = None
    
    try:
        # 1. Check if it's a local file path (absolute or relative)
        if os.path.exists(input_spec):
            input_type = "local_file"
            print(f"Detected local file: {input_spec}")
            
            structure_name = os.path.splitext(os.path.basename(input_spec))[0]
            
            if input_spec.lower().endswith('.pdb'):
                print("Processing PDB file...")
                entry = pdb_to_dict_format(input_spec, structure_name)
                return entry, temp_files
            elif input_spec.lower().endswith('.cif'):
                print("Processing CIF file...")
                entry = cif_to_dict_format(input_spec, structure_name)
                return entry, temp_files
            else:
                raise ValueError(f"Unsupported file format: {input_spec}. Supported: .pdb, .cif")
        
        # 2. Parse as PDB ID (with optional chain)
        if '.' in input_spec:
            pdb_id, chain_id = input_spec.split('.', 1)
            input_type = "pdb_id_with_chain"
            if verbose:
                print(f"Detected PDB ID with chain: {pdb_id}.{chain_id}")
        else:
            pdb_id = input_spec
            chain_id = None
            input_type = "pdb_id"
            if verbose:
                print(f"Detected PDB ID: {pdb_id}")
        
        # 3. Validate PDB ID format (4 characters, alphanumeric)
        if not (len(pdb_id) == 4 and pdb_id.isalnum()):
            raise ValueError(
                f"Invalid PDB ID format: '{pdb_id}'. Expected 4 alphanumeric characters.\n"
                f"Examples: '1abc', '2XYZ', '1fcd.C' (with chain)"
            )
        
        # 4. Download PDB file
        temp_dir = tempfile.mkdtemp()
        if verbose:
            print(f"Creating temporary directory: {temp_dir}")

        pdb_path = download_pdb(pdb_id, os.path.join(temp_dir, f"{pdb_id.lower()}.pdb"))
        temp_files.append(pdb_path)
        
        # 5. Extract specific chain if requested
        if chain_id:
            print(f"Extracting chain {chain_id}...")
            pdb_path = extract_chain_from_pdb(pdb_path, chain_id)
            temp_files.append(pdb_path)
            structure_name = f"{pdb_id.upper()}.{chain_id.upper()}"
        else:
            structure_name = pdb_id.upper()
        
        # 6. Convert to dict format
        if verbose:
            print("Converting to internal dictionary format...")
        entry = pdb_to_dict_format(pdb_path, structure_name)
        
        print(f"Successfully processed {input_type}: {structure_name}")
        return entry, temp_files
        
    except Exception as e:
        # Cleanup temp files on error
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        
        # Provide helpful error message based on input type
        if input_type == "local_file":
            raise ValueError(f"Failed to process local file '{input_spec}': {e}")
        elif input_type in ["pdb_id", "pdb_id_with_chain"]:
            raise ValueError(f"Failed to process PDB ID '{input_spec}': {e}\n"
                           f"Make sure the PDB ID exists and is accessible from RCSB PDB database.")
        else:
            raise ValueError(f"Failed to process input '{input_spec}': {e}\n"
                           f"Supported formats:\n"
                           f"  - Local files: '/path/to/file.pdb', 'protein.cif'\n" 
                           f"  - PDB IDs: '1abc', '2XYZ'\n"
                           f"  - PDB ID + chain: '1fcd.C', '2abc.A'")



class CustomInputDataset:
    """Minimal dataset for custom input compatibility with existing sampling functions."""
    def __init__(self, graph_data, entry):
        self.graph_data = graph_data
        self.entry = entry
        
    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("Only one entry")
        return self.graph_data, None, None, 0.0, None  # data, y_true, mask, time, dssp
        
    def __len__(self):
        return 1


def compute_ensemble_consensus(protein_states, consensus_strength, ensemble_method='arithmetic', ensemble_size=None):
    """
    Compute ensemble consensus using either arithmetic or geometric averaging.
    
    Args:
        protein_states: [ensemble_size, seq_len, K] tensor of probability states
        consensus_strength: Float between 0-1, strength of consensus blending
        ensemble_method: 'arithmetic' or 'geometric'
        ensemble_size: Size of ensemble (auto-detected if None)
        
    Returns:
        Blended protein states with same shape as input, normalized on simplex
    """
    if ensemble_size is None:
        ensemble_size = protein_states.shape[0]
    
    if ensemble_method == 'arithmetic':
        # Standard arithmetic mean in probability space
        protein_mean = protein_states.mean(dim=0, keepdim=True)  # [1, seq_len, K]
        
        # Apply consensus blending
        blended_states = (1 - consensus_strength) * protein_states + \
                        consensus_strength * protein_mean.expand(ensemble_size, -1, -1)
    
    elif ensemble_method == 'geometric':
        # Geometric mean in log space
        eps = 1e-8  # Small epsilon to avoid log(0)
        
        # Convert to log space (add epsilon to avoid log(0))
        log_states = torch.log(protein_states + eps)  # [ensemble_size, seq_len, K]
        
        # Compute mean in log space
        log_mean = log_states.mean(dim=0, keepdim=True)  # [1, seq_len, K]
        
        # Apply consensus blending in log space
        blended_log_states = (1 - consensus_strength) * log_states + \
                            consensus_strength * log_mean.expand(ensemble_size, -1, -1)
        
        # Convert back to probability space
        blended_states = torch.exp(blended_log_states)
    
    else:
        raise ValueError(f"Unknown ensemble_method: {ensemble_method}. Choose 'arithmetic' or 'geometric'")
    
    # Ensure states remain on simplex (normalize to sum to 1)
    return simplex_proj(blended_states)


def compute_sampling_metrics(predicted_probabilities, ground_truth_onehot, data, model, args, device, use_virtual_node, K):
    """
    Compute comprehensive evaluation metrics for sampling results.
    
    Args:
        predicted_probabilities: [N, K] tensor of predicted probabilities
        ground_truth_onehot: [N, K] tensor of ground truth one-hot vectors
        data: Graph data object
        model: Trained model
        args: Arguments object
        device: Device to run computations on
        use_virtual_node: Whether virtual nodes are used
        K: Number of amino acid classes
        
    Returns:
        Dict containing evaluation metrics
    """
    with torch.no_grad():
        # Ensure tensors are on the same device
        if predicted_probabilities.device != device:
            predicted_probabilities = predicted_probabilities.to(device)
        if ground_truth_onehot.device != device:
            ground_truth_onehot = ground_truth_onehot.to(device)
        
        # Basic accuracy
        predicted_classes = predicted_probabilities.argmax(-1)
        true_classes = ground_truth_onehot.argmax(-1)
        
        # Create mask to exclude unknown residues (X = index 20) from accuracy calculation
        valid_mask = (true_classes != 20)  # Exclude positions where ground truth is X/unknown
        
        if valid_mask.sum() > 0:
            # Only calculate accuracy for non-unknown positions
            correct = (predicted_classes == true_classes).float()
            accuracy = correct[valid_mask].mean().item()
            
            # Cross-entropy loss with hard labels (only for valid positions)
            cce_loss_hard = F.cross_entropy(predicted_probabilities[valid_mask], true_classes[valid_mask]).item()
            
            # Cross-entropy loss with soft labels (only for valid positions)
            eps = 1e-8
            log_probs = torch.log(predicted_probabilities + eps)
            cce_loss_smooth = -(ground_truth_onehot[valid_mask] * log_probs[valid_mask]).sum(-1).mean().item()
            
            # Per-position confidence (only for valid positions)
            confidence = predicted_probabilities[valid_mask].max(-1)[0].mean().item()
            
            # Entropy of predictions (only for valid positions)
            entropy = -(predicted_probabilities[valid_mask] * torch.log(predicted_probabilities[valid_mask] + eps)).sum(-1).mean().item()
            
            # Top-k accuracy (k=3, only for valid positions)
            top3_predictions = predicted_probabilities[valid_mask].topk(3, dim=-1)[1]
            top3_correct = (top3_predictions == true_classes[valid_mask].unsqueeze(-1)).any(-1).float()
            top3_accuracy = top3_correct.mean().item()
        else:
            # If all positions are unknown, set default values
            accuracy = 0.0
            cce_loss_hard = float('inf')
            cce_loss_smooth = float('inf')
            confidence = 0.0
            entropy = 0.0
            top3_accuracy = 0.0
        
        # Per-class accuracy (excluding unknown residues)
        per_class_correct = torch.zeros(K, device=device)
        per_class_total = torch.zeros(K, device=device)
        
        if valid_mask.sum() > 0:
            correct = (predicted_classes == true_classes).float()
            for k in range(K):
                if k == 20:  # Skip unknown class (X/XXX)
                    continue
                mask = (true_classes == k) & valid_mask  # Only count valid positions
                if mask.sum() > 0:
                    per_class_correct[k] = correct[mask].sum()
                    per_class_total[k] = mask.sum()
        
        eps = 1e-8
        per_class_accuracy = per_class_correct / (per_class_total + eps)
        
        metrics = {
            'accuracy': accuracy,
            'cce_loss_hard': cce_loss_hard,
            'cce_loss_smooth': cce_loss_smooth,
            'confidence': confidence,
            'entropy': entropy,
            'top3_accuracy': top3_accuracy,
            'per_class_accuracy': per_class_accuracy.cpu().numpy(),
            'total_positions': len(true_classes),
            'valid_positions': valid_mask.sum().item(),
            'unknown_positions': (true_classes == 20).sum().item(),
            'fraction_valid': valid_mask.sum().item() / len(true_classes) if len(true_classes) > 0 else 0.0
        }
        
        return metrics


def simplex_proj(seq):
    """
    Project sequences onto the probability simplex using Euclidean projection.
    
    This ensures that each position's probability distribution sums to 1 and
    all probabilities are non-negative, which maintains the proper simplex constraint.
    
    Args:
        seq: Tensor of shape [B, N, K] representing probability distributions
        
    Returns:
        Projected tensor of the same shape
    """
    # Handle negative values first by clamping to 0
    seq_pos = torch.clamp(seq, min=0.0)
    
    # Normalize to ensure sum equals 1 for each position
    seq_sum = seq_pos.sum(-1, keepdim=True)
    
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    seq_normalized = seq_pos / (seq_sum + eps)
    
    return seq_normalized


class DistributedSamplingDataset(Dataset):
    """
    Dataset wrapper for distributed sampling that provides index-based access
    to structures for sampling across multiple GPUs.
    """
    
    def __init__(self, base_dataset, indices=None):
        """
        Initialize distributed sampling dataset.
        
        Args:
            base_dataset: Base dataset (e.g., CathDataset)
            indices: List of indices to include (None for all)
        """
        self.base_dataset = base_dataset
        self.indices = indices if indices is not None else list(range(len(base_dataset)))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx], actual_idx


class SamplingCoordinator:
    """
    Coordinates distributed sampling across multiple GPUs with proper cleanup.
    """
    
    def __init__(self, model_path: str, dataset_path: str, split: str = 'validation'):
        """
        Initialize sampling coordinator.
        
        Args:
            model_path: Path to trained model
            dataset_path: Path to dataset
            split: Dataset split to use
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.split = split
        self.is_distributed = False
        self.model = None
        self.dataset = None
    
    def setup_distributed(self, rank: int, world_size: int, master_port: int = 29500):
        """Setup distributed training environment."""
        import torch.distributed as dist
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(master_port)
        
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )
        
        torch.cuda.set_device(rank)
        self.is_distributed = True
    
    def cleanup_distributed(self):
        """Cleanup distributed environment."""
        if self.is_distributed:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
            self.is_distributed = False
    
    def load_model_and_dataset(self, device: torch.device, args=None):
        """Load model and dataset on specified device."""
        # Load model and extract dataset parameters
        self.model, dataset_params = load_model_distributed(self.model_path, device, args)
        
        # Store dataset_params for access by other methods
        self.dataset_params = dataset_params
        
        # Use command-line overrides if provided, otherwise use dataset parameters from checkpoint
        split_json = (args.split_json if args and args.split_json else dataset_params.get('split_json')) or '../datasets/cath-4.2/chain_set_splits.json'
        map_pkl = (args.map_pkl if args and args.map_pkl else dataset_params.get('map_pkl')) or '../datasets/cath-4.2/chain_set_map_with_b_factors.pkl'
        use_virtual_node = dataset_params.get('use_virtual_node', True)
        
        # Extract RBF parameters from dataset_params (should always be available now)
        rbf_3d_min = dataset_params.get('rbf_3d_min')
        rbf_3d_max = dataset_params.get('rbf_3d_max')
        rbf_3d_spacing = dataset_params.get('rbf_3d_spacing')
        
        if rbf_3d_min is None or rbf_3d_max is None or rbf_3d_spacing is None:
            raise RuntimeError(
                f"Internal error: RBF parameters should have been resolved in load_model_distributed.\n"
                f"Got: rbf_3d_min={rbf_3d_min}, rbf_3d_max={rbf_3d_max}, rbf_3d_spacing={rbf_3d_spacing}"
            )
        
        print(f"Using dataset parameters:")
        print(f"  split_json: {split_json}" + (" (command-line override)" if args and args.split_json else " (from checkpoint)" if dataset_params.get('split_json') else " (default)"))
        print(f"  map_pkl: {map_pkl}" + (" (command-line override)" if args and args.map_pkl else " (from checkpoint)" if dataset_params.get('map_pkl') else " (default)"))
        print(f"  use_virtual_node: {use_virtual_node}")
        print(f"  rbf_3d_min: {rbf_3d_min}")
        print(f"  rbf_3d_max: {rbf_3d_max}")
        print(f"  rbf_3d_spacing: {rbf_3d_spacing}")
        
        # Prepare graph builder parameters
        graph_builder_kwargs = {
            'k': dataset_params.get('k_neighbors'),
            'k_farthest': dataset_params.get('k_farthest'),
            'k_random': dataset_params.get('k_random'),
            'max_edge_dist': dataset_params.get('max_edge_dist'),
            'num_rbf_3d': dataset_params.get('num_rbf_3d'),
            'num_rbf_seq': dataset_params.get('num_rbf_seq'),
            'no_source_indicator': dataset_params.get('no_source_indicator'),
            'rbf_3d_min': rbf_3d_min,
            'rbf_3d_max': rbf_3d_max,
            'rbf_3d_spacing': rbf_3d_spacing,
        }
        
        # Special handling for GraphBuilder parameter validation:
        # When max_edge_dist is specified, k/k_farthest/k_random should be explicitly None
        # When max_edge_dist is None, k/k_farthest/k_random must be provided
        if dataset_params.get('max_edge_dist') is not None:
            # Distance-based mode: ensure k parameters are explicitly None
            graph_builder_kwargs.update({
                'k': None,
                'k_farthest': None,
                'k_random': None,
            })
        
        # Remove None values to use GraphBuilder defaults (except for distance-based mode)
        if dataset_params.get('max_edge_dist') is not None:
            # In distance-based mode, keep None values for k parameters
            graph_builder_kwargs = {k: v for k, v in graph_builder_kwargs.items() 
                                  if v is not None or k in ['k', 'k_farthest', 'k_random']}
        else:
            # In k-neighbor mode, remove None values 
            graph_builder_kwargs = {k: v for k, v in graph_builder_kwargs.items() if v is not None}
        
        # Load dataset
        from data.cath_dataset import CathDataset
        self.dataset = CathDataset(
            split_json=split_json,
            map_pkl=map_pkl,
            split=self.split,
            graph_builder_kwargs=graph_builder_kwargs,
            # Add required time sampling parameters with defaults for sampling
            time_sampling_strategy='uniform',
            t_min=0.0,
            t_max=8.0,
            alpha_range=1.0
        )
        
        return self.model, self.dataset
    
    def sample_structures(self, indices: List[int], device: torch.device, args=None) -> List[Dict]:
        """Sample sequences for given structure indices."""
        if self.model is None or self.dataset is None:
            raise RuntimeError("Model and dataset must be loaded first")
        
        from .sample import sample_multiple_proteins
        
        return sample_multiple_proteins(
            self.model, 
            self.dataset, 
            indices=indices,
            args=args
        )


def setup_distributed_sampling(device='auto'):
    """
    Setup distributed sampling environment.
    
    Args:
        device: Device specification ('auto', 'cuda', 'cpu', or specific device)
        
    Returns:
        Tuple of (device, world_size, rank)
    """
    import torch.distributed as dist
    
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Check if distributed is available and initialized
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if not dist.is_available():
            print("Warning: Distributed training not available, falling back to single GPU")
            return device, 1, 0
        
        if not dist.is_initialized():
            print("Warning: Distributed not initialized, falling back to single GPU")
            return device, 1, 0
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        print(f"Distributed sampling: rank {rank}/{world_size} on device {device}")
        return device, world_size, rank
    else:
        print(f"Single device sampling on {device}")
        return device, 1, 0


def cleanup_distributed_sampling():
    """Cleanup distributed sampling environment."""
    import torch.distributed as dist
    
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _download_checkpoint_from_url(url: str, cache_dir: str = "/tmp/checkpoint_cache") -> str:
    """
    Download a checkpoint from a URL with caching support.
    Supports Azure CLI, urllib (built-in), and requests as fallback.
    
    Args:
        url: The URL to download from (Azure blob, HTTP, etc.)
        cache_dir: Directory to cache downloaded checkpoints
        
    Returns:
        str: Path to the downloaded checkpoint file
    """
    import hashlib
    import os
    import subprocess
    from pathlib import Path
    from urllib.parse import urlparse
    
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Generate cache filename from URL hash
    url_hash = hashlib.md5(url.encode()).hexdigest()
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path) or "checkpoint.pt"
    cache_filename = f"{url_hash}_{filename}"
    cached_file_path = cache_path / cache_filename
    
    # Return cached file if it exists
    if cached_file_path.exists():
        print(f"Using cached checkpoint: {cached_file_path}")
        return str(cached_file_path)
    
    print(f"Downloading checkpoint from: {url}")
    print(f"Caching to: {cached_file_path}")
    
    # Method 1: Try Azure CLI first (most reliable for Azure blobs)
    if "blob.core.windows.net" in url:
        try:
            print("Attempting download with Azure CLI...")
            result = subprocess.run([
                "az", "storage", "blob", "download", 
                "--blob-url", url,
                "--file", str(cached_file_path)
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                print(f"Successfully downloaded checkpoint using Azure CLI: {cached_file_path}")
                return str(cached_file_path)
            else:
                print(f"Azure CLI download failed: {result.stderr}")
                print("Falling back to other methods...")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Azure CLI not available or timeout: {e}")
            print("Falling back to other methods...")
    
    # Method 2: Try urllib (built-in Python, no extra dependencies)
    try:
        import urllib.request
        import urllib.error
        
        print("Attempting download with urllib (built-in)...")
        
        # Download with progress reporting for large files
        def download_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) / total_size)
                if block_num % 1000 == 0:  # Print every ~8MB for typical block sizes
                    print(f"Download progress: {percent:.1f}%")
        
        urllib.request.urlretrieve(url, cached_file_path, reporthook=download_progress)
        print(f"Successfully downloaded checkpoint using urllib: {cached_file_path}")
        return str(cached_file_path)
        
    except urllib.error.URLError as e:
        print(f"urllib download failed: {e}")
    except Exception as e:
        print(f"urllib download failed with unexpected error: {e}")
    
    # Method 3: Try requests as final fallback (if available)
    try:
        import requests
        print("Attempting download with requests (fallback)...")
        
        # Download with streaming to handle large files
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size if available
        file_size = int(response.headers.get('content-length', 0))
        if file_size > 0:
            print(f"Download size: {file_size / (1024*1024):.1f} MB")
        
        # Download and save
        with open(cached_file_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if file_size > 0 and downloaded % (1024*1024*10) == 0:  # Progress every 10MB
                        progress = (downloaded / file_size) * 100
                        print(f"Download progress: {progress:.1f}%")
        
        print(f"Successfully downloaded checkpoint using requests: {cached_file_path}")
        return str(cached_file_path)
        
    except ImportError:
        print("requests library not available")
    except Exception as e:
        print(f"requests download failed: {e}")
    
    # Clean up partial download and raise error
    if cached_file_path.exists():
        cached_file_path.unlink()
    
    raise RuntimeError(f"Failed to download checkpoint from {url} using all available methods (Azure CLI, urllib, requests)")


def _auto_discover_best_model(original_path: str) -> str:
    """
    Auto-discover local 'best' model files in the workspace.
    
    Searches in multiple locations for files containing 'best' and ending with '.pt':
    1. Current working directory
    2. Parent directory of the original path
    3. Common model directories (./saved_models, ./output/saved_models, etc.)
    
    Args:
        original_path: The original path that wasn't found
        
    Returns:
        str: Path to the discovered model file, or None if not found
    """
    import glob
    import os
    from pathlib import Path
    
    print(f"Auto-discovering local 'best' model files (original path not found: {original_path})")
    
    # Get search directories
    search_dirs = []
    
    # 1. Current working directory
    search_dirs.append(os.getcwd())
    
    # 2. Parent directory of original path (if it has one)
    if original_path and os.path.dirname(original_path):
        search_dirs.append(os.path.dirname(original_path))
    
    # 3. Common model directory patterns
    common_dirs = [
        './saved_models',
        './output/saved_models',
        '../output/saved_models',
        './models',
        './checkpoints',
        '.'  # Current directory as fallback
    ]
    search_dirs.extend(common_dirs)
    
    # Remove duplicates and ensure directories exist
    search_dirs = [d for d in set(search_dirs) if os.path.isdir(d)]
    
    print(f"Searching in directories: {search_dirs}")
    
    best_models = []
    
    # Search for files containing 'best' and ending with '.pt'
    for search_dir in search_dirs:
        pattern = os.path.join(search_dir, '*best*.pt')
        matches = glob.glob(pattern)
        if matches:
            best_models.extend(matches)
            print(f"Found {len(matches)} 'best' models in {search_dir}: {[os.path.basename(m) for m in matches]}")
    
    if not best_models:
        print("No local 'best' model files found")
        return None
    
    # Remove duplicates and sort by modification time (newest first)
    best_models = list(set(best_models))
    best_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    selected_model = best_models[0]
    print(f"Auto-selected newest 'best' model: {selected_model}")
    
    if len(best_models) > 1:
        print(f"Note: Found {len(best_models)} 'best' models, using newest: {os.path.basename(selected_model)}")
        print("Other candidates:")
        for model in best_models[1:]:
            mtime = os.path.getmtime(model)
            print(f"  - {os.path.basename(model)} (modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))})")
    
    return selected_model


def load_model_distributed(model_path: str, device: torch.device, args):
    """
    Load a trained model for distributed sampling and extract dataset parameters.
    Supports both local file paths and Azure blob URLs.
    
    Args:
        model_path: Path to the model checkpoint (local path or Azure URL)
        device: Device to load the model on
        args: Arguments object with model configuration
        
    Returns:
        Tuple of (model, dataset_params) where dataset_params contains
        the dataset configuration extracted from the checkpoint
    """
    # Dynamic import to avoid circular dependencies
    import sys
    import os
    
    # Add the parent directory to the Python path if not already there
    parent_dir = os.path.join(os.path.dirname(__file__), '..')
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    from models.dfm_model import DFMNodeClassifier
    
    print(f"Loading model from: {model_path}")
    
    # Handle Azure URLs or remote URLs
    local_model_path = model_path
    if model_path.startswith('http://') or model_path.startswith('https://'):
        local_model_path = _download_checkpoint_from_url(model_path)
    elif not os.path.exists(model_path):
        # Try to auto-discover local 'best' model files
        auto_discovered_path = _auto_discover_best_model(model_path)
        if auto_discovered_path:
            print(f"Auto-discovered local best model: {auto_discovered_path}")
            local_model_path = auto_discovered_path
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(local_model_path, map_location=device)
    
    # Extract model arguments
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        print("="*60)
        print("CHECKPOINT PARAMETER EXTRACTION")
        print("="*60)
        print("Found 'args' in checkpoint - extracting model configuration")
        
        # Determine if model_args is a dict or an object
        is_dict = isinstance(model_args, dict)
        print(f"Args type: {type(model_args).__name__}")
        
        # Helper function to get values from either dict or object
        def get_arg_value(key, default=None):
            if is_dict:
                return model_args.get(key, default)
            else:
                return getattr(model_args, key, default)
        
        # Show all available parameters
        print("\nAll available parameters in checkpoint args:")
        if is_dict:
            for key in sorted(model_args.keys()):
                value = model_args[key]
                print(f"  {key}: {value}")
        elif hasattr(model_args, '__dict__'):
            for attr_name in sorted(model_args.__dict__.keys()):
                attr_value = getattr(model_args, attr_name)
                print(f"  {attr_name}: {attr_value}")
        else:
            print("  Warning: model_args has no accessible attributes")
        
        print("\nKey model architecture parameters:")
        print(f"  num_layers_gvp: {get_arg_value('num_layers_gvp', get_arg_value('num_layers', 'NOT FOUND'))}")  # Fallback for old checkpoints
        print(f"  num_layers_prediction: {get_arg_value('num_layers_prediction', 'NOT FOUND')}")
        print(f"  hidden_dim: {get_arg_value('hidden_dim', 'NOT FOUND')}")
        print(f"  use_qkv: {get_arg_value('use_qkv', 'NOT FOUND')}")
        print(f"  time_dim: {get_arg_value('time_dim', 'NOT FOUND')}")
        print(f"  use_virtual_node: {get_arg_value('use_virtual_node', 'NOT FOUND')}")
        print(f"  node_dims: {(get_arg_value('node_dim_s'), get_arg_value('node_dim_v'))}")
        print(f"  edge_dims: {(get_arg_value('edge_dim_s'), get_arg_value('edge_dim_v'))}")
        print(f"  hidden_dims: {(get_arg_value('hidden_dim'), get_arg_value('hidden_dim_v'))}")
        print(f"  dropout: {get_arg_value('dropout', 'NOT FOUND')}")
        
        print("\nDataset-related parameters:")
        
        # Check for dedicated graph_builder_params section first (new format)
        graph_builder_params = checkpoint.get('graph_builder_params', {})
        if graph_builder_params:
            print("Using dataset parameters from dedicated graph_builder_params section")
            split_json_cp = graph_builder_params.get('split_json') or get_arg_value('split_json')
            map_pkl_cp = graph_builder_params.get('map_pkl') or get_arg_value('map_pkl')
            use_virtual_cp = graph_builder_params.get('use_virtual_node')
            if use_virtual_cp is None:
                use_virtual_cp = get_arg_value('use_virtual_node')
        else:
            print("Using dataset parameters from args section")
            split_json_cp = get_arg_value('split_json')
            map_pkl_cp = get_arg_value('map_pkl')
            use_virtual_cp = get_arg_value('use_virtual_node')
            
        max_length_cp = get_arg_value('max_length')
        use_graph_builder_cp = get_arg_value('use_graph_builder')
        
        print(f"  split_json: {split_json_cp}")
        print(f"  map_pkl: {map_pkl_cp}")
        print(f"  use_virtual_node: {use_virtual_cp}")
        print(f"  max_length: {max_length_cp}")
        print(f"  use_graph_builder: {use_graph_builder_cp}")
        
        print("\nGraph building parameters:")
        
        # Check for dedicated graph_builder_params section first (new format)
        graph_builder_params = checkpoint.get('graph_builder_params', {})
        if graph_builder_params:
            print("Found dedicated graph_builder_params section in checkpoint")
            # Handle parameter name mapping (GraphBuilder uses 'k' but training uses 'k_neighbors')
            k_neighbors_cp = graph_builder_params.get('k') or graph_builder_params.get('k_neighbors')
            k_farthest_cp = graph_builder_params.get('k_farthest')
            k_random_cp = graph_builder_params.get('k_random')
            max_edge_dist_cp = graph_builder_params.get('max_edge_dist')
            num_rbf_3d_cp = graph_builder_params.get('num_rbf_3d')
            num_rbf_seq_cp = graph_builder_params.get('num_rbf_seq')
            no_source_indicator_cp = graph_builder_params.get('no_source_indicator')
            # RBF distance range parameters (new)
            rbf_3d_min_cp = graph_builder_params.get('rbf_3d_min')
            rbf_3d_max_cp = graph_builder_params.get('rbf_3d_max')
            rbf_3d_spacing_cp = graph_builder_params.get('rbf_3d_spacing')
        else:
            print("No dedicated graph_builder_params section found, extracting from args")
            k_neighbors_cp = get_arg_value('k_neighbors')
            k_farthest_cp = get_arg_value('k_farthest')
            k_random_cp = get_arg_value('k_random')
            max_edge_dist_cp = get_arg_value('max_edge_dist')
            num_rbf_3d_cp = get_arg_value('num_rbf_3d')
            num_rbf_seq_cp = get_arg_value('num_rbf_seq')
            no_source_indicator_cp = get_arg_value('no_source_indicator')
            # RBF distance range parameters (fallback to defaults if not in args)
            rbf_3d_min_cp = get_arg_value('rbf_3d_min')
            rbf_3d_max_cp = get_arg_value('rbf_3d_max')
            rbf_3d_spacing_cp = get_arg_value('rbf_3d_spacing')
        
        # Check for model architecture parameters (new systematic format)
        model_architecture_params = checkpoint.get('model_architecture_params', {})
        if model_architecture_params:
            print("Found dedicated model_architecture_params section in checkpoint")
            print("Model architecture parameters from checkpoint:")
            for param_name, param_value in model_architecture_params.items():
                print(f"  {param_name}: {param_value}")
        else:
            print("No dedicated model_architecture_params section found")
        
        print(f"  k_neighbors: {k_neighbors_cp}")
        print(f"  k_farthest: {k_farthest_cp}")
        print(f"  k_random: {k_random_cp}")
        print(f"  max_edge_dist: {max_edge_dist_cp}")
        print(f"  num_rbf_3d: {num_rbf_3d_cp}")
        print(f"  num_rbf_seq: {num_rbf_seq_cp}")
        print(f"  no_source_indicator: {no_source_indicator_cp}")
        print(f"  rbf_3d_min: {rbf_3d_min_cp} (default: 2.0 if None)")
        print(f"  rbf_3d_max: {rbf_3d_max_cp} (default: 350.0 if None)")
        print(f"  rbf_3d_spacing: {rbf_3d_spacing_cp} (default: exponential if None)")
        
        print("\nTraining parameters:")
        print(f"  learning_rate (lr): {get_arg_value('lr', 'NOT FOUND')}")
        print(f"  batch_size (batch): {get_arg_value('batch', 'NOT FOUND')}")
        print(f"  epochs: {get_arg_value('epochs', 'NOT FOUND')}")
        print(f"  alpha_min: {get_arg_value('alpha_min', 'NOT FOUND')}")
        print(f"  alpha_max: {get_arg_value('alpha_max', 'NOT FOUND')}")
        
        print("\nTime parameters:")
        t_max_cp = get_arg_value('t_max')
        t_min_cp = get_arg_value('t_min')
        print(f"  t_max: {t_max_cp} (default: 8.0 if None)")
        print(f"  t_min: {t_min_cp} (default: 0.0 if None)")
        
        # Extract dataset parameters with detailed fallback logic
        print("\n" + "="*60)
        print("DATASET PARAMETER RESOLUTION")
        print("="*60)
        
        # Always use command line arguments if available, then fall back to checkpoint, then default
        # For split_json
        if hasattr(args, 'split_json') and args.split_json:
            final_split_json = args.split_json
            split_json_source = "command line args"
        elif split_json_cp:
            final_split_json = split_json_cp
            split_json_source = "checkpoint"
        else:
            final_split_json = '../datasets/cath-4.2/chain_set_splits.json'
            split_json_source = "default"
        
        # For map_pkl
        if hasattr(args, 'map_pkl') and args.map_pkl:
            final_map_pkl = args.map_pkl
            map_pkl_source = "command line args"
        elif map_pkl_cp:
            final_map_pkl = map_pkl_cp
            map_pkl_source = "checkpoint"
        else:
            final_map_pkl = '../datasets/cath-4.2/chain_set_map_with_b_factors.pkl'
            map_pkl_source = "default"
        
        # For use_virtual_node
        if use_virtual_cp is not None:
            final_use_virtual = use_virtual_cp
            virtual_source = "checkpoint"
        else:
            final_use_virtual = None  # Will be inferred later
            virtual_source = "will be inferred from filename"
        
        dataset_params = {
            'split_json': final_split_json,
            'map_pkl': final_map_pkl,
            'use_virtual_node': final_use_virtual,
            'max_length': max_length_cp,
            'use_graph_builder': use_graph_builder_cp if use_graph_builder_cp is not None else True,
            # Graph builder parameters from checkpoint
            'k_neighbors': k_neighbors_cp,
            'k_farthest': k_farthest_cp,
            'k_random': k_random_cp,
            'max_edge_dist': max_edge_dist_cp,
            'num_rbf_3d': num_rbf_3d_cp,
            'num_rbf_seq': num_rbf_seq_cp,
            'no_source_indicator': no_source_indicator_cp,
        }
        
        # Handle RBF distance range parameters with backwards compatibility
        # Priority: checkpoint > command line args > error (no defaults for backwards compatibility)
        rbf_3d_min_final = None
        rbf_3d_max_final = None
        rbf_3d_spacing_final = None
        rbf_source = None
        
        if rbf_3d_min_cp is not None and rbf_3d_max_cp is not None and rbf_3d_spacing_cp is not None:
            # All RBF parameters found in checkpoint
            rbf_3d_min_final = rbf_3d_min_cp
            rbf_3d_max_final = rbf_3d_max_cp
            rbf_3d_spacing_final = rbf_3d_spacing_cp
            rbf_source = "checkpoint"
        elif args and hasattr(args, 'rbf_3d_min') and hasattr(args, 'rbf_3d_max') and hasattr(args, 'rbf_3d_spacing'):
            # Check if command line args are provided
            if args.rbf_3d_min is not None and args.rbf_3d_max is not None and args.rbf_3d_spacing is not None:
                rbf_3d_min_final = args.rbf_3d_min
                rbf_3d_max_final = args.rbf_3d_max
                rbf_3d_spacing_final = args.rbf_3d_spacing
                rbf_source = "command line arguments"
            else:
                # Some command line args are None
                missing_args = []
                if args.rbf_3d_min is None:
                    missing_args.append('--rbf_3d_min')
                if args.rbf_3d_max is None:
                    missing_args.append('--rbf_3d_max')
                if args.rbf_3d_spacing is None:
                    missing_args.append('--rbf_3d_spacing')
                
                raise ValueError(
                    f"RBF parameters not found in checkpoint and missing from command line.\n"
                    f"For backwards compatibility with older checkpoints, please provide:\n"
                    f"  {' '.join(missing_args)}\n"
                    f"Example: --rbf_3d_min 2.0 --rbf_3d_max 350.0 --rbf_3d_spacing exponential\n"
                    f"Checkpoint RBF params: rbf_3d_min={rbf_3d_min_cp}, rbf_3d_max={rbf_3d_max_cp}, rbf_3d_spacing={rbf_3d_spacing_cp}"
                )
        else:
            # No args object or missing attributes
            raise ValueError(
                f"RBF parameters not found in checkpoint and no command line arguments available.\n"
                f"For backwards compatibility with older checkpoints, please provide:\n"
                f"  --rbf_3d_min <value> --rbf_3d_max <value> --rbf_3d_spacing <spacing_type>\n"
                f"Example: --rbf_3d_min 2.0 --rbf_3d_max 350.0 --rbf_3d_spacing exponential\n"
                f"Checkpoint RBF params: rbf_3d_min={rbf_3d_min_cp}, rbf_3d_max={rbf_3d_max_cp}, rbf_3d_spacing={rbf_3d_spacing_cp}"
            )
        
        # Add RBF parameters to dataset_params
        dataset_params['rbf_3d_min'] = rbf_3d_min_final
        dataset_params['rbf_3d_max'] = rbf_3d_max_final
        dataset_params['rbf_3d_spacing'] = rbf_3d_spacing_final
        
        # Add model architecture parameters from checkpoint
        dataset_params['model_architecture_params'] = model_architecture_params
        
        # Add time parameters to dataset_params
        dataset_params['t_max'] = t_max_cp or 8.0  # Default to 8.0 if not in checkpoint
        dataset_params['t_min'] = t_min_cp or 0.0  # Default to 0.0 if not in checkpoint
        
        print(f"split_json: {final_split_json} (source: {split_json_source})")
        print(f"map_pkl: {final_map_pkl} (source: {map_pkl_source})")
        print(f"use_virtual_node: {final_use_virtual} (source: {virtual_source})")
        print(f"max_length: {max_length_cp} (source: {'checkpoint' if max_length_cp else 'not specified'})")
        print(f"use_graph_builder: {dataset_params['use_graph_builder']} (source: {'checkpoint' if use_graph_builder_cp else 'default'})")
        
        print(f"\nGraph builder parameter resolution:")
        print(f"k_neighbors: {k_neighbors_cp} (source: {'checkpoint' if k_neighbors_cp is not None else 'not in checkpoint'})")
        print(f"k_farthest: {k_farthest_cp} (source: {'checkpoint' if k_farthest_cp is not None else 'not in checkpoint'})")
        print(f"k_random: {k_random_cp} (source: {'checkpoint' if k_random_cp is not None else 'not in checkpoint'})")
        print(f"max_edge_dist: {max_edge_dist_cp} (source: {'checkpoint' if max_edge_dist_cp is not None else 'not in checkpoint'})")
        print(f"num_rbf_3d: {num_rbf_3d_cp} (source: {'checkpoint' if num_rbf_3d_cp is not None else 'not in checkpoint'})")
        print(f"num_rbf_seq: {num_rbf_seq_cp} (source: {'checkpoint' if num_rbf_seq_cp is not None else 'not in checkpoint'})")
        
        print(f"\nRBF distance range parameters:")
        print(f"rbf_3d_min: {rbf_3d_min_final} (source: {rbf_source})")
        print(f"rbf_3d_max: {rbf_3d_max_final} (source: {rbf_source})")
        print(f"rbf_3d_spacing: {rbf_3d_spacing_final} (source: {rbf_source})")
        
        print(f"\nTime parameter resolution:")
        print(f"t_max: {dataset_params['t_max']} (source: {'checkpoint' if t_max_cp else 'default'})")
        print(f"t_min: {dataset_params['t_min']} (source: {'checkpoint' if t_min_cp else 'default'})")
            
    else:
        raise Exception(f'Checkpoint arguments are not found for model in {model_path}')
        
    
    # Infer architecture from state_dict keys if args are missing
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE INFERENCE")
    print("="*60)
    
    # Find maximum layer number from state_dict keys
    max_layer = 0
    layer_keys = []
    interleaved_layer_keys = []
    
    # Check for embed layers
    for key in state_dict.keys():
        if 'gnn.gnn.embed.' in key:
            try:
                layer_num = int(key.split('gnn.gnn.embed.')[1].split('.')[0])
                max_layer = max(max_layer, layer_num)
                layer_keys.append((layer_num, key))
            except:
                pass
    
    # Check for interleaved layers
    for key in state_dict.keys():
        if 'gnn.gnn.interleaved_layers.' in key:
            try:
                layer_num = int(key.split('gnn.gnn.interleaved_layers.')[1].split('.')[0])
                max_layer = max(max_layer, layer_num)
                interleaved_layer_keys.append((layer_num, key))
            except:
                pass
    
    # Infer num_layers from state dict
    inferred_num_layers = max_layer + 1  # layers are 0-indexed
    
    print(f"State dict analysis:")
    print(f"  Total keys in state_dict: {len(state_dict.keys())}")
    print(f"  Embed layer keys found: {len(layer_keys)}")
    print(f"  Interleaved layer keys found: {len(interleaved_layer_keys)}")
    if layer_keys:
        print(f"  Embed layer range: 0 to {max([ln for ln, _ in layer_keys])}")
        print(f"  Example embed layer keys: {[key for _, key in sorted(layer_keys)[:3]]}")
    if interleaved_layer_keys:
        print(f"  Interleaved layer range: 0 to {max([ln for ln, _ in interleaved_layer_keys])}")
        print(f"  Example interleaved layer keys: {[key for _, key in sorted(interleaved_layer_keys)[:3]]}")
    if layer_keys and len(layer_keys) > 3:
        print(f"    ... and {len(layer_keys) - 3} more embed keys")
    if interleaved_layer_keys and len(interleaved_layer_keys) > 3:
        print(f"    ... and {len(interleaved_layer_keys) - 3} more interleaved keys")
        if len(layer_keys) > 3:
            print(f"    ... and {len(layer_keys) - 3} more")
    print(f"  Inferred num_layers: {inferred_num_layers}")
    
    # Check for virtual node usage from model filename and dataset params
    model_filename = os.path.basename(model_path)
    use_virtual_inferred = 'noVirtual' not in model_filename and 'Novirtual' not in model_filename
    
    print(f"\nVirtual node inference:")
    print(f"  Model filename: {model_filename}")
    print(f"  Contains 'noVirtual': {'noVirtual' in model_filename}")
    print(f"  Contains 'Novirtual': {'Novirtual' in model_filename}")
    print(f"  Inferred use_virtual_node: {use_virtual_inferred}")
    
    # Use dataset parameter if available, otherwise use inference from filename
    if dataset_params['use_virtual_node'] is not None:
        use_virtual_final = dataset_params['use_virtual_node']
        print(f"  Final use_virtual_node: {use_virtual_final} (from checkpoint)")
    else:
        use_virtual_final = use_virtual_inferred
        print(f"  Final use_virtual_node: {use_virtual_final} (inferred from filename)")
        # Update dataset_params with inferred value
        dataset_params['use_virtual_node'] = use_virtual_final
    
    # Create model with proper parameter structure, using inferred values when needed
    print("\n" + "="*60)
    print("FINAL MODEL CREATION PARAMETERS")
    print("="*60)
    
    # Detect architecture details from processed checkpoint keys
    checkpoint_keys = set(state_dict.keys())  # Use processed state_dict (without module. prefix)
    has_old_layers = any('gnn.gnn.layers.' in key for key in checkpoint_keys)
    has_new_message_layers = any('gnn.gnn.message_layers.' in key for key in checkpoint_keys)
    has_new_embed = any('gnn.gnn.embed.' in key for key in checkpoint_keys)
    has_interleaved = any('gnn.gnn.interleaved_layers.' in key for key in checkpoint_keys)
    
    print(f"Architecture detection:")
    print(f"  Checkpoint has old 'layers': {has_old_layers}")
    print(f"  Checkpoint has new 'message_layers': {has_new_message_layers}")
    print(f"  Checkpoint has new 'embed': {has_new_embed}")
    print(f"  Checkpoint has 'interleaved_layers': {has_interleaved}")
    
    # Use the EXACT architecture from the checkpoint args first
    checkpoint_architecture = get_arg_value('architecture', 'blocked')
    print(f"  Checkpoint specifies architecture: '{checkpoint_architecture}'")
    
    # Auto-detect architecture parameters from checkpoint structure
    if has_interleaved:
        # Interleaved architecture detected - count actual layers from state dict
        interleaved_layer_indices = set()
        for key in checkpoint_keys:
            if 'gnn.gnn.interleaved_layers.' in key:
                try:
                    layer_idx = int(key.split('gnn.gnn.interleaved_layers.')[1].split('.')[0])
                    interleaved_layer_indices.add(layer_idx)
                except:
                    pass
        
        total_interleaved_layers = len(interleaved_layer_indices)
        max_interleaved_idx = max(interleaved_layer_indices) if interleaved_layer_indices else 0
        print(f"Detected interleaved architecture with {total_interleaved_layers} total layers (indices: {sorted(interleaved_layer_indices)})")
        
        # For interleaved, extract the actual embed/message layer counts from checkpoint args
        # But ensure they match the actual structure in the state dict
        checkpoint_embed_layers = get_arg_value('num_layers_gvp', get_arg_value('num_layers', 5))
        checkpoint_message_layers = get_arg_value('num_message_layers', 2)
        
        # Validate against state dict structure - the total should match
        expected_total = checkpoint_embed_layers + checkpoint_message_layers
        if total_interleaved_layers != expected_total:
            print(f"  Warning: Checkpoint args indicate {expected_total} layers ({checkpoint_embed_layers} embed + {checkpoint_message_layers} message)")
            print(f"           but state dict has {total_interleaved_layers} interleaved layers")
            print(f"  Using actual state dict structure for model creation")
            
            # Infer layer distribution from the actual structure
            # Typically embed layers come first, then message layers
            # But we need to create a model that matches the checkpoint exactly
            inferred_num_embed_layers = checkpoint_embed_layers
            inferred_num_message_layers = checkpoint_message_layers
        else:
            inferred_num_embed_layers = checkpoint_embed_layers
            inferred_num_message_layers = checkpoint_message_layers
            
        inferred_architecture = 'interleaved'
        use_legacy_naming = False
        
        print(f"Using interleaved architecture from checkpoint:")
        print(f"  num_embed_layers: {inferred_num_embed_layers}")
        print(f"  num_message_layers: {inferred_num_message_layers}")
        print(f"  architecture: {inferred_architecture}")
        
    elif has_old_layers and not has_new_message_layers:
        # Old architecture - need to determine layer split
        old_layer_indices = set()
        for key in checkpoint_keys:
            if key.startswith('gnn.gnn.layers.'):
                try:
                    layer_idx = int(key.split('gnn.gnn.layers.')[1].split('.')[0])
                    old_layer_indices.add(layer_idx)
                except:
                    pass
        
        total_old_layers = len(old_layer_indices)
        print(f"Detected old architecture with {total_old_layers} total layers")
        
        # For full compatibility, create a model that uses the old 'layers' naming
        # by setting num_message_layers=0 so ResidueGNN uses the old architecture
        inferred_num_embed_layers = total_old_layers
        inferred_num_message_layers = 0  # Force old architecture
        inferred_architecture = 'blocked'
        use_legacy_naming = True
        
        print(f"Using legacy-compatible architecture:")
        print(f"  num_embed_layers: {inferred_num_embed_layers}")
        print(f"  num_message_layers: {inferred_num_message_layers} (forces old 'layers' naming)")
        print(f"  architecture: {inferred_architecture}")
        
    else:
        # New blocked architecture or mixed - use checkpoint parameters exactly
        inferred_num_embed_layers = get_arg_value('num_layers_gvp', get_arg_value('num_layers', 4))
        inferred_num_message_layers = get_arg_value('num_message_layers', 1)
        inferred_architecture = checkpoint_architecture  # Use exact architecture from checkpoint
        use_legacy_naming = False
        
        print(f"Using modern blocked architecture:")
        print(f"  num_embed_layers: {inferred_num_embed_layers}")
        print(f"  num_message_layers: {inferred_num_message_layers}")
        print(f"  architecture: {inferred_architecture}")
    
    gvp_kwargs = {
        'node_dims': (get_arg_value('node_dim_s', 10), get_arg_value('node_dim_v', 3)),
        'edge_dims': (get_arg_value('edge_dim_s', 32), get_arg_value('edge_dim_v', 1)),
        'hidden_dims': (get_arg_value('hidden_dim', 256), get_arg_value('hidden_dim_v', 64)),
        'num_layers': inferred_num_embed_layers,  # For backward compatibility
        'num_embed_layers': inferred_num_embed_layers,
        'num_message_layers': inferred_num_message_layers,
        'architecture': inferred_architecture,
        'use_qkv': get_arg_value('use_qkv', True),
        'dropout': get_arg_value('dropout', 0.1),
        # use_virtual_node is handled at the data level, not ResidueGNN parameter
    }
    
    dfm_kwargs = {
        'K': 21,
        'alpha_min': get_arg_value('alpha_min', 1.0),
        'alpha_max': get_arg_value('alpha_max', 10.0),
        'alpha_spacing': get_arg_value('alpha_spacing', 0.01)
    }
    
    print("GVP kwargs (for GNN architecture):")
    for key, value in gvp_kwargs.items():
        if key in ['node_dims', 'edge_dims', 'hidden_dims']:
            source = "checkpoint" if all(get_arg_value(f"{key.split('_')[0]}_{dim}") is not None for dim in ['s', 'v']) else "default/inferred"
        else:
            source = "checkpoint" if get_arg_value(key) is not None else "default/inferred"
        print(f"  {key}: {value} (source: {source})")
    
    print("\nDFM kwargs (for flow matching):")
    for key, value in dfm_kwargs.items():
        source = "checkpoint" if get_arg_value(key) is not None else "default"
        print(f"  {key}: {value} (source: {source})")
    
    print(f"\nAdditional model parameters:")
    time_dim = get_arg_value('time_dim', 64)
    time_scale = get_arg_value('time_scale', 1.0)
    head_hidden = get_arg_value('head_hidden', 256)
    head_dropout = get_arg_value('head_dropout', 0.1)
    head_depth = get_arg_value('num_layers_prediction', 4)
    recycle_steps = get_arg_value('recycle_steps', 1)
    time_integration = get_arg_value('time_integration', 'film')
    use_time_conditioning = not get_arg_value('disable_time_conditioning', False)
    
    print(f"  time_dim: {time_dim} (source: {'checkpoint' if get_arg_value('time_dim') is not None else 'default'})")
    print(f"  time_scale: {time_scale} (source: {'checkpoint' if get_arg_value('time_scale') is not None else 'default'})")
    print(f"  head_hidden: {head_hidden} (source: {'checkpoint' if get_arg_value('head_hidden') is not None else 'default'})")
    print(f"  head_dropout: {head_dropout} (source: {'checkpoint' if get_arg_value('head_dropout') is not None else 'default'})")
    print(f"  head_depth: {head_depth} (source: {'checkpoint' if get_arg_value('num_layers_prediction') is not None else 'default'})")
    print(f"  recycle_steps: {recycle_steps} (source: {'checkpoint' if get_arg_value('recycle_steps') is not None else 'default'})")
    print(f"  time_integration: {time_integration} (source: {'checkpoint' if get_arg_value('time_integration') is not None else 'default'})")
    print(f"  use_time_conditioning: {use_time_conditioning} (source: {'checkpoint' if get_arg_value('disable_time_conditioning') is not None else 'default'})")
    
    # Extract DSSP loss parameter
    lambda_dssp_loss = get_arg_value('lambda_dssp_loss')
    
    print(f"\nCreating DFMNodeClassifier with {inferred_num_embed_layers} embed layers, {inferred_num_message_layers} message layers, {head_depth} prediction head layers")
    print(f"Architecture: {inferred_architecture}, virtual_node={use_virtual_final} (stored in dataset_params)")
    if lambda_dssp_loss is not None and lambda_dssp_loss > 0:
        print(f"DSSP multitask learning enabled: lambda_dssp_loss={lambda_dssp_loss}")
    else:
        print("DSSP multitask learning disabled")
    
    model = DFMNodeClassifier(
        gvp_kwargs=gvp_kwargs,
        dfm_kwargs=dfm_kwargs,
        time_dim=time_dim,
        time_scale=time_scale,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
        head_depth=head_depth,
        recycle_steps=recycle_steps,
        time_integration=time_integration,
        use_time_conditioning=use_time_conditioning,
        lambda_dssp_loss=lambda_dssp_loss
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Preprocess state dict to handle DataParallel/DistributedDataParallel keys
    # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
    processed_state_dict = {}
    has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
    
    if has_module_prefix:
        print("Detected DataParallel/DistributedDataParallel checkpoint - removing 'module.' prefix")
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix
                processed_state_dict[new_key] = value
            else:
                processed_state_dict[key] = value
        state_dict = processed_state_dict
        print(f"Preprocessed state dict: {len(state_dict)} keys after prefix removal")
    
    # Architecture detection and key adaptation
    print("\n" + "="*60)
    print("ARCHITECTURE DETECTION AND KEY ADAPTATION")
    print("="*60)
    
    # Detect architecture from checkpoint keys
    checkpoint_keys = set(state_dict.keys())
    model_keys = set(model.state_dict().keys())
    
    # Detect if checkpoint uses old 'layers' naming vs new 'message_layers' + 'embed' naming
    has_old_layers = any('gnn.gnn.layers.' in key for key in checkpoint_keys)
    has_new_message_layers = any('gnn.gnn.message_layers.' in key for key in checkpoint_keys)
    has_new_embed = any('gnn.gnn.embed.' in key for key in checkpoint_keys)
    has_interleaved_layers = any('gnn.gnn.interleaved_layers.' in key for key in checkpoint_keys)
    
    # Detect architecture type from model keys
    model_has_message_layers = any('gnn.gnn.message_layers.' in key for key in model_keys)
    model_has_embed = any('gnn.gnn.embed.' in key for key in model_keys)
    model_has_interleaved = any('gnn.gnn.interleaved_layers.' in key for key in model_keys)
    
    print(f"Checkpoint architecture detection:")
    print(f"  Has old 'layers' naming: {has_old_layers}")
    print(f"  Has new 'message_layers': {has_new_message_layers}")
    print(f"  Has new 'embed': {has_new_embed}")
    print(f"  Has 'interleaved_layers': {has_interleaved_layers}")
    
    print(f"Model architecture detection:")
    print(f"  Has 'message_layers': {model_has_message_layers}")
    print(f"  Has 'embed': {model_has_embed}")
    print(f"  Has 'interleaved_layers': {model_has_interleaved}")
    
    # Check for fundamental architecture mismatches that cannot be resolved by key mapping
    if has_interleaved_layers and not model_has_interleaved:
        print("\nCRITICAL ERROR: Architecture Mismatch!")
        print("  Checkpoint uses 'interleaved_layers' architecture")
        print("  Model was created with 'blocked' architecture")
        print("  This requires recreating the model with architecture='interleaved'")
        print("\nThe model must be recreated with the EXACT same architecture as the checkpoint.")
        raise RuntimeError("Architecture mismatch: checkpoint is 'interleaved' but model is 'blocked'. Cannot load.")
    
    if not has_interleaved_layers and model_has_interleaved:
        print("\nCRITICAL ERROR: Architecture Mismatch!")
        print("  Checkpoint uses 'blocked' architecture") 
        print("  Model was created with 'interleaved' architecture")
        print("  This requires recreating the model with architecture='blocked'")
        print("\nThe model must be recreated with the EXACT same architecture as the checkpoint.")
        raise RuntimeError("Architecture mismatch: checkpoint is 'blocked' but model is 'interleaved'. Cannot load.")
    
    # Determine if we need key mapping
    needs_key_mapping = False
    mapping_strategy = None
    
    if has_old_layers and (model_has_message_layers or model_has_embed or model_has_interleaved):
        needs_key_mapping = True
        if model_has_interleaved:
            mapping_strategy = "old_to_interleaved"
        elif model_has_message_layers and model_has_embed:
            mapping_strategy = "old_to_blocked"
        else:
            mapping_strategy = "old_to_unknown"
    elif (has_new_message_layers or has_new_embed) and model_has_interleaved:
        needs_key_mapping = True
        mapping_strategy = "blocked_to_interleaved"
    elif (has_new_message_layers or has_new_embed) and (model_has_message_layers and model_has_embed):
        needs_key_mapping = False  # Same architecture
        mapping_strategy = "no_mapping_needed"
    
    print(f"Key mapping needed: {needs_key_mapping}")
    print(f"Mapping strategy: {mapping_strategy}")
    
    if needs_key_mapping:
        print("\nApplying flexible key mapping...")
        adapted_state_dict = {}
        
        if mapping_strategy == "old_to_blocked":
            # Map old 'layers' to new 'embed' structure only - ignore message_layers for now
            # This creates a simpler mapping that should work better
            
            # First, copy all non-GNN keys as-is
            for key, value in state_dict.items():
                if not key.startswith('gnn.gnn.layers.'):
                    adapted_state_dict[key] = value
            
            # Extract layer information from old format
            old_layer_keys = [key for key in checkpoint_keys if key.startswith('gnn.gnn.layers.')]
            old_layer_indices = set()
            for key in old_layer_keys:
                try:
                    layer_idx = int(key.split('gnn.gnn.layers.')[1].split('.')[0])
                    old_layer_indices.add(layer_idx)
                except:
                    pass
            
            old_layer_indices = sorted(old_layer_indices)
            print(f"  Found {len(old_layer_indices)} old layers: {old_layer_indices}")
            
            # Simple direct mapping: old layers -> embed layers with same indices
            print(f"  Direct mapping: old layers -> embed layers")
            mapped_count = 0
            for old_idx in old_layer_indices:
                for key in old_layer_keys:
                    if key.startswith(f'gnn.gnn.layers.{old_idx}.'):
                        suffix = key[len(f'gnn.gnn.layers.{old_idx}.'):]
                        new_key = f'gnn.gnn.embed.{old_idx}.{suffix}'
                        adapted_state_dict[new_key] = state_dict[key]
                        mapped_count += 1
                        if mapped_count <= 5:  # Show first few mappings
                            print(f"    Mapped: {key} -> {new_key}")
            
            if mapped_count > 5:
                print(f"    ... and {mapped_count - 5} more mappings")
            
            print(f"  Successfully mapped {mapped_count} parameter keys")
        
        elif mapping_strategy == "old_to_interleaved":
            # Map old 'layers' to new 'interleaved_layers' structure
            print("  Mapping old layers to interleaved architecture...")
            
            # Copy all non-GNN keys as-is
            for key, value in state_dict.items():
                if not key.startswith('gnn.gnn.layers.'):
                    adapted_state_dict[key] = value
            
            # Map old layers to interleaved layers sequentially
            old_layer_keys = [key for key in checkpoint_keys if key.startswith('gnn.gnn.layers.')]
            old_layer_indices = set()
            for key in old_layer_keys:
                try:
                    layer_idx = int(key.split('gnn.gnn.layers.')[1].split('.')[0])
                    old_layer_indices.add(layer_idx)
                except:
                    pass
            
            old_layer_indices = sorted(old_layer_indices)
            
            # Get interleaved layer count from model
            model_interleaved_indices = set()
            for key in model_keys:
                if key.startswith('gnn.gnn.interleaved_layers.'):
                    try:
                        layer_idx = int(key.split('gnn.gnn.interleaved_layers.')[1].split('.')[0])
                        model_interleaved_indices.add(layer_idx)
                    except:
                        pass
            
            model_interleaved_indices = sorted(model_interleaved_indices)
            print(f"  Model interleaved layers: {model_interleaved_indices}")
            
            # Map old layers to interleaved layers sequentially
            for i, old_idx in enumerate(old_layer_indices):
                if i < len(model_interleaved_indices):
                    new_idx = model_interleaved_indices[i]
                    for key in old_layer_keys:
                        if key.startswith(f'gnn.gnn.layers.{old_idx}.'):
                            suffix = key[len(f'gnn.gnn.layers.{old_idx}.'):]
                            new_key = f'gnn.gnn.interleaved_layers.{new_idx}.{suffix}'
                            adapted_state_dict[new_key] = state_dict[key]
                            print(f"    Mapped: {key} -> {new_key}")
        
        else:
            print(f"  Warning: Mapping strategy '{mapping_strategy}' not implemented, attempting partial mapping...")
            # Copy everything we can and skip what we can't
            for key, value in state_dict.items():
                if key in model_keys:
                    adapted_state_dict[key] = value
        
        # Use adapted state dict
        final_state_dict = adapted_state_dict
        print(f"Successfully created adapted state dict with {len(final_state_dict)} keys")
        
    else:
        print("No key mapping needed - architectures are compatible")
        final_state_dict = state_dict
    
    # Final validation
    model_keys = set(model.state_dict().keys())
    final_checkpoint_keys = set(final_state_dict.keys())
    
    missing_keys = model_keys - final_checkpoint_keys
    unexpected_keys = final_checkpoint_keys - model_keys
    
    print(f"\nFinal validation after key adaptation:")
    print(f"  Missing keys: {len(missing_keys)}")
    print(f"  Unexpected keys: {len(unexpected_keys)}")
    
    if missing_keys:
        print(f"  Missing keys (first 5): {list(missing_keys)[:5]}")
    if unexpected_keys:
        print(f"  Unexpected keys (first 5): {list(unexpected_keys)[:5]}")
    
    # Only fail if we have significant missing keys (not just a few parameters)
    critical_missing_threshold = 0.1  # Allow up to 10% missing keys
    critical_missing_ratio = len(missing_keys) / len(model_keys) if model_keys else 0
    
    if critical_missing_ratio > critical_missing_threshold:
        print(f"\nFATAL ERROR: Too many missing keys after adaptation!")
        print(f"Missing {len(missing_keys)}/{len(model_keys)} keys ({critical_missing_ratio:.1%} > {critical_missing_threshold:.1%} threshold)")
        print(f"This suggests fundamental architecture incompatibility that cannot be resolved by key mapping.")
        
        raise RuntimeError(f"Model loading failed: {critical_missing_ratio:.1%} of model keys are missing from checkpoint")
    
    # Load with strict=False to handle minor remaining key differences
    load_result = model.load_state_dict(final_state_dict, strict=False)
    
    if load_result.missing_keys:
        print(f"\nWarning: {len(load_result.missing_keys)} parameters could not be loaded from checkpoint:")
        for key in load_result.missing_keys[:5]:
            print(f"  - {key}")
        if len(load_result.missing_keys) > 5:
            print(f"  ... and {len(load_result.missing_keys) - 5} more")
        print("These parameters will be randomly initialized.")
    
    if load_result.unexpected_keys:
        print(f"\nInfo: {len(load_result.unexpected_keys)} parameters in checkpoint were not used:")
        for key in load_result.unexpected_keys[:5]:
            print(f"  - {key}")
        if len(load_result.unexpected_keys) > 5:
            print(f"  ... and {len(load_result.unexpected_keys) - 5} more")
        print("These parameters were ignored.")
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"\nModel loaded successfully on {device}")
    
    print("\n" + "="*60)
    print("FINAL DATASET PARAMETERS TO BE USED")
    print("="*60)
    print("These parameters will be used for dataset creation:")
    for key, value in dataset_params.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    return model, dataset_params


def run_sampling_and_evaluation(checkpoint_path: str, args, device, output_base: str, model_name: str, job_timestamp: str):
    """
    Run sampling and evaluation using an external checkpoint.
    This function is called when running in sampling-only mode.
    
    Args:
        checkpoint_path: Path or URL to the checkpoint
        args: Training arguments object
        device: PyTorch device
        output_base: Base output directory
        model_name: Name for the model
        job_timestamp: Timestamp for this job
        
    Returns:
        dict: Results information including output directory
    """
    import os
    import json
    from pathlib import Path
    from data.cath_dataset import CathDataset
    
    print(f"\nLoading model from checkpoint: {checkpoint_path}")
    
    # Load model and dataset parameters from checkpoint
    model, dataset_params = load_model_distributed(checkpoint_path, device, args)
    
    # Create dataset for sampling using parameters from checkpoint
    print(f"\nCreating dataset with parameters from checkpoint:")
    for key, value in dataset_params.items():
        if value is not None:
            print(f"  {key}: {value}")
    
    # Prepare graph builder parameters
    graph_builder_kwargs = {
        'k': dataset_params.get('k_neighbors'),
        'k_farthest': dataset_params.get('k_farthest'),
        'k_random': dataset_params.get('k_random'),
        'num_rbf_3d': dataset_params.get('num_rbf_3d'),
        'num_rbf_seq': dataset_params.get('num_rbf_seq'),
        'no_source_indicator': dataset_params.get('no_source_indicator'),
        # RBF distance range parameters (should always be available now)
        'rbf_3d_min': dataset_params.get('rbf_3d_min'),
        'rbf_3d_max': dataset_params.get('rbf_3d_max'),
        'rbf_3d_spacing': dataset_params.get('rbf_3d_spacing')
    }
    
    # Validate RBF parameters
    if (graph_builder_kwargs['rbf_3d_min'] is None or 
        graph_builder_kwargs['rbf_3d_max'] is None or 
        graph_builder_kwargs['rbf_3d_spacing'] is None):
        raise RuntimeError(
            f"Internal error: RBF parameters should have been resolved in load_model_distributed.\n"
            f"Got: rbf_3d_min={graph_builder_kwargs['rbf_3d_min']}, "
            f"rbf_3d_max={graph_builder_kwargs['rbf_3d_max']}, "
            f"rbf_3d_spacing={graph_builder_kwargs['rbf_3d_spacing']}"
        )
    
    # Remove None values to use GraphBuilder defaults
    graph_builder_kwargs = {k: v for k, v in graph_builder_kwargs.items() if v is not None}
    
    # Create CATH dataset for sampling
    dataset = CathDataset(
        split_json=dataset_params['split_json'],
        map_pkl=dataset_params['map_pkl'],
        split='validation',  # Default to validation set for sampling
        max_len=dataset_params.get('max_length'),
        graph_builder_kwargs=graph_builder_kwargs,
        # Required time sampling parameters for sampling
        time_sampling_strategy='uniform',
        t_min=0.0,
        t_max=8.0,
        alpha_range=1.0
    )
    
    print(f"Dataset created with {len(dataset)} structures")
    
    # Set up output directories
    output_dir = os.path.join(output_base, 'sampling_results', job_timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Run sampling with parameters from config or defaults
    sampling_config = getattr(args, 'sampling', {}) if hasattr(args, 'sampling') else {}
    
    # Extract sampling parameters (with defaults)
    num_samples = sampling_config.get('num_samples', 100)
    steps = sampling_config.get('steps', 20)
    T = sampling_config.get('T', 8.0)
    t_min = sampling_config.get('t_min', 0.0)
    temperature = sampling_config.get('flow_temp', 1.0)
    integration_method = sampling_config.get('integration_method', 'euler')
    save_probabilities = sampling_config.get('save_probabilities', True)
    
    # Get indices to sample (first num_samples from validation set)
    indices = list(range(min(num_samples, len(dataset))))
    
    print(f"\nStarting sampling with parameters:")
    print(f"  Number of samples: {len(indices)}")
    print(f"  Steps: {steps}")
    print(f"  Temperature: {temperature}")
    print(f"  Integration method: {integration_method}")
    print(f"  T: {T}, t_min: {t_min}")
    
    # Run sampling with batching for efficiency
    batch_size = getattr(args, 'batch_size', 4)  # Default batch size of 4
    results = sample_multiple_proteins_batched(
        model=model,
        dataset=dataset,
        indices=indices,
        steps=steps,
        T=T,
        t_min=t_min,
        save_probabilities=save_probabilities,
        integration_method=integration_method,
        batch_size=batch_size,
        args=args
    )
    
    # Save results
    output_prefix = f"{model_name}_{job_timestamp}"
    save_results_to_files(
        results=results,
        output_prefix=output_prefix,
        output_dir=output_dir,
        model_name=model_name,
        split='validation',
        steps=steps,
        T=T
    )
    
    print(f"\nSampling completed! Results saved to: {output_dir}")
    
    # Optional: Run structure prediction and evaluation if configured
    # This would integrate with your evaluation pipeline
    
    return {
        'output_dir': output_dir,
        'num_samples': len(indices),
        'results': results
    }


def align_sequence_to_structure(full_sequence, structure_sequence, verbose=False):
    """
    Efficiently align a full sequence (e.g., from UniProt) to a structure sequence.
    
    Args:
        full_sequence: Full sequence string
        structure_sequence: Structure sequence string (what's in the dataset)
        verbose: Print alignment details
        
    Returns:
        Dict with alignment information including start position and mapping
    """
    if verbose:
        print(f"Aligning sequences:")
        print(f"  Full sequence length: {len(full_sequence)}")
        print(f"  Structure sequence length: {len(structure_sequence)}")
    
    if not full_sequence or not structure_sequence:
        raise ValueError("Both full_sequence and structure_sequence must be non-empty")
    
    if len(structure_sequence) > len(full_sequence):
        raise ValueError(f"Structure sequence ({len(structure_sequence)}) cannot be longer than full sequence ({len(full_sequence)})")
    
    # Simple substring search for exact matches - O(n*m) but very fast for exact matches
    start_pos = full_sequence.find(structure_sequence)
    
    if start_pos != -1:
        # Perfect match
        if verbose:
            print(f"  Perfect match found at position {start_pos}")
        
        # Create position mapping: structure_pos -> full_seq_pos
        mapping = {i: start_pos + i for i in range(len(structure_sequence))}
        
        return {
            'mapping': mapping,
            'start_pos': start_pos,
            'alignment_score': 1.0
        }
    
    # Fuzzy matching with early termination
    best_score = 0
    best_start = 0
    min_required_score = 0.8
    struct_len = len(structure_sequence)
    min_matches_required = int(struct_len * min_required_score)
    
    # Sliding window approach with optimizations
    search_end = len(full_sequence) - struct_len + 1
    for start in range(search_end):
        matches = 0
        max_possible_remaining = struct_len
        
        # Count matches with early termination
        for i in range(struct_len):
            if structure_sequence[i] == full_sequence[start + i]:
                matches += 1
            
            max_possible_remaining -= 1
            # Early termination: if we can't possibly beat the best score, skip
            if matches + max_possible_remaining <= best_score * struct_len:
                break
            
            # Early success: if we already have enough matches, we can stop counting
            if matches >= min_matches_required:
                matches = matches  # Continue counting for exact score
                for j in range(i + 1, struct_len):
                    if structure_sequence[j] == full_sequence[start + j]:
                        matches += 1
                break
        
        score = matches / struct_len
        if score > best_score:
            best_score = score
            best_start = start
            
            # Early termination: if we found a very good match, stop searching
            if score >= 0.95:
                break
    
    if best_score >= min_required_score:
        best_mapping = {i: best_start + i for i in range(struct_len)}
        
        if verbose:
            print(f"  Best fuzzy match: {best_score:.2%} at position {best_start}")
            
        return {
            'mapping': best_mapping,
            'start_pos': best_start,
            'alignment_score': best_score
        }
    else:
        if verbose:
            print(f"  No good alignment found (best score: {best_score:.2%})")
        
        return {
            'mapping': {},
            'start_pos': -1,
            'alignment_score': best_score
        }


def create_inpainting_mask_with_alignment(full_sequence, structure_sequence, 
                                         mask_positions=None, known_sequence=None, 
                                         mask_ratio=0.3, verbose=False, device='cpu'):
    """
    Create an inpainting mask using alignment between full and structure sequences.
    
    Args:
        full_sequence: Full sequence (e.g., from UniProt)
        structure_sequence: Structure sequence (from dataset)
        mask_positions: Positions in FULL sequence to mask
        known_sequence: Known template with 'X' for positions to predict
        mask_ratio: Random masking ratio if no specific positions
        verbose: Print detailed information
        device: Device for PyTorch tensor
        
    Returns:
        Dict with mask information and alignment details (mask as PyTorch tensor)
    """
    # First align the sequences
    alignment_info = align_sequence_to_structure(full_sequence, structure_sequence, verbose)
    
    if alignment_info['alignment_score'] < 0.8:
        raise ValueError(f"Poor sequence alignment (score: {alignment_info['alignment_score']:.3f} < 0.8). "
                        f"Cannot reliably map positions between sequences. "
                        f"Full sequence length: {len(full_sequence)}, "
                        f"Structure sequence length: {len(structure_sequence)}")
    
    structure_len = len(structure_sequence)
    mask = torch.zeros(structure_len, dtype=torch.bool, device=device)
    
    if mask_positions is not None:
        # Convert full sequence positions to structure positions
        if isinstance(mask_positions, str):
            positions = [int(x.strip()) for x in mask_positions.split(',')]
        else:
            positions = mask_positions
        
        mapping = alignment_info['mapping']
        reverse_mapping = {v: k for k, v in mapping.items()}
        
        masked_count = 0
        for full_pos in positions:
            if full_pos in reverse_mapping:
                struct_pos = reverse_mapping[full_pos]
                mask[struct_pos] = True
                masked_count += 1
                if verbose:
                    print(f"  Full position {full_pos} -> Structure position {struct_pos}")
        
        if verbose:
            print(f"Mapped {masked_count}/{len(positions)} positions from full to structure sequence")
    
    elif known_sequence is not None:
        # Use known sequence template
        if len(known_sequence) != len(full_sequence):
            raise ValueError(f"Known sequence length ({len(known_sequence)}) doesn't match full sequence length ({len(full_sequence)})")
        
        mapping = alignment_info['mapping']
        masked_count = 0
        
        for struct_pos, full_pos in mapping.items():
            if full_pos < len(known_sequence) and known_sequence[full_pos].upper() == 'X':
                mask[struct_pos] = True
                masked_count += 1
        
        if verbose:
            print(f"Mapped {masked_count} masked positions from known sequence template")
    
    else:
        # Random masking
        num_to_mask = int(structure_len * mask_ratio)
        if num_to_mask > 0:
            random_indices = torch.randperm(structure_len, device=device)[:num_to_mask]
            mask[random_indices] = True
        
        if verbose:
            print(f"Random masking: {num_to_mask} positions")
    
    return {
        'mask': mask,
        'alignment_info': alignment_info,
        'num_masked': mask.sum().item()
    }


def create_simple_inpainting_mask(structure_sequence, mask_positions=None, 
                                known_sequence=None, mask_ratio=0.3, device='cpu'):
    """
    Create a simple inpainting mask without sequence alignment.
    
    Args:
        structure_sequence: Structure sequence string
        mask_positions: Specific positions to mask (0-indexed)
        known_sequence: Template with 'X' for positions to predict
        mask_ratio: Random masking ratio
        device: Device for PyTorch tensor
        
    Returns:
        Dict with mask information (mask as PyTorch tensor)
    """
    structure_len = len(structure_sequence)
    mask = torch.zeros(structure_len, dtype=torch.bool, device=device)
    
    if mask_positions is not None:
        if isinstance(mask_positions, str):
            positions = [int(x.strip()) for x in mask_positions.split(',')]
        else:
            positions = mask_positions
        
        for pos in positions:
            if 0 <= pos < structure_len:
                mask[pos] = True
    
    elif known_sequence is not None:
        if len(known_sequence) != structure_len:
            raise ValueError(f"Known sequence length ({len(known_sequence)}) doesn't match structure sequence length ({structure_len})")
        
        for i, aa in enumerate(known_sequence):
            if aa.upper() == 'X':
                mask[i] = True
    
    else:
        # Random masking
        num_to_mask = int(structure_len * mask_ratio)
        if num_to_mask > 0:
            random_indices = torch.randperm(structure_len, device=device)[:num_to_mask]
            mask[random_indices] = True
    
    return {
        'mask': mask,
        'alignment_info': {'mapping': None, 'start_pos': None, 'alignment_score': 0.0},
        'num_masked': mask.sum().item()
    }


def generate_detailed_json_output(results, structure_names, output_dir, output_prefix, K=21):
    """
    Generate detailed JSON output with time-step information for each protein.
    
    Args:
        results: List of result dictionaries containing trajectory data
        structure_names: List of structure names/PDB IDs
        output_dir: Output directory
        output_prefix: Prefix for output filename
        
    Returns:
        str: Path to the generated JSON file
    """
    import numpy as np
    
    # Build output data structure
    detailed_output = {}
    
    for i, (result, structure_name) in enumerate(zip(results, structure_names)):
        if 'trajectory_data' not in result:
            print(f"Warning: No trajectory data for structure {structure_name}")
            continue
        
        trajectory = result['trajectory_data']
        pdb_id = structure_name if structure_name != 'unknown' else f"structure_{i}"
        
        # Initialize structure data
        detailed_output[pdb_id] = {}
        
        # Process each position
        for pos, pos_data in trajectory['positions'].items():
            # Get trajectory for this position
            time_points = pos_data['time_points']
            most_likely_aas = pos_data['most_likely_aa']
            probabilities = pos_data['probabilities']
            detailed_breakdowns = pos_data.get('detailed_breakdown', [])
            
            # Create trajectory entries with detailed amino acid breakdown
            trajectory_entries = []
            for idx, (t, aa_idx, prob) in enumerate(zip(time_points, most_likely_aas, probabilities)):
                # Bounds checking for most likely AA
                if 0 <= aa_idx < len(IDX_TO_AA):
                    aa_name = IDX_TO_AA[aa_idx]
                    aa_single = THREE_TO_ONE.get(aa_name, 'X')
                else:
                    aa_name = 'XXX'
                    aa_single = 'X'
                
                trajectory_entry = {
                    'time_point': round(float(t), 4),
                    'most_likely_amino_acid': aa_single,
                    'amino_acid_name': aa_name,
                    'current_probability': round(float(prob), 6)
                }
                
                # Add detailed amino acid breakdown if available
                if idx < len(detailed_breakdowns):
                    trajectory_entry['amino_acid_breakdown'] = detailed_breakdowns[idx]
                
                trajectory_entries.append(trajectory_entry)
            
            # Get ground truth information for this position if available
            ground_truth_aa = None
            ground_truth_name = None
            
            if 'true_indices' in result and result['true_indices'] is not None:
                true_indices = result['true_indices']
                if pos < len(true_indices):
                    true_idx = true_indices[pos]
                    if 0 <= true_idx < len(IDX_TO_AA):
                        ground_truth_name = IDX_TO_AA[true_idx]
                        ground_truth_aa = THREE_TO_ONE.get(ground_truth_name, 'X')
                    else:
                        ground_truth_name = 'XXX'
                        ground_truth_aa = 'X'
            
            # Store position data
            position_info = {
                'trajectory': trajectory_entries,
                'final_prediction': trajectory_entries[-1]['most_likely_amino_acid'] if trajectory_entries else 'X',
                'final_probability': trajectory_entries[-1]['current_probability'] if trajectory_entries else 0.0,
                'ground_truth': ground_truth_aa,
                'ground_truth_name': ground_truth_name
            }
            
            detailed_output[pdb_id][str(pos)] = position_info
    
    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_filename = f"{timestamp}_{output_prefix}_detailed_predictions.json"
    json_filepath = os.path.join(output_dir, json_filename)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON with pretty formatting
    with open(json_filepath, 'w') as f:
        json.dump(detailed_output, f, indent=2, sort_keys=True)
    
    print(f"Detailed JSON output saved to: {json_filepath}")
    
    # Generate trajectory analysis file for get_prediction_accuracy function
    trajectory_analysis_data = {}
    
    for i, (result, structure_name) in enumerate(zip(results, structure_names)):
        if 'trajectory_data' not in result:
            continue
            
        trajectory = result['trajectory_data']
        pdb_id = structure_name if structure_name != 'unknown' else f"structure_{i}"
        
        # Extract model predictions at each timestep
        time_points = trajectory['time_points']
        positions = trajectory['positions']
        
        if not positions:
            continue
            
        # Get sequence length from first position
        first_pos_key = list(positions.keys())[0]
        first_pos_data = positions[first_pos_key]
        num_timesteps = len(first_pos_data['time_points'])
        
        # Initialize arrays for this protein
        sequence_length = len(positions)
        model_predictions = []
        current_states = []
        
        # Extract data for each timestep
        for timestep in range(num_timesteps):
            timestep_predictions = np.zeros((sequence_length, K))
            timestep_states = np.zeros((sequence_length, K))
            
            for pos_idx, (pos_key, pos_data) in enumerate(positions.items()):
                if timestep < len(pos_data.get('detailed_breakdown', [])):
                    # Get the full probability distribution for this timestep
                    breakdown = pos_data['detailed_breakdown'][timestep]
                    for aa_single_letter, aa_data in breakdown.items():
                        # Convert single-letter AA to 3-letter, then to index
                        if aa_single_letter in SINGLE_TO_TRIPLE:
                            aa_three_letter = SINGLE_TO_TRIPLE[aa_single_letter]
                            if aa_three_letter in AA_TO_IDX:
                                aa_idx = AA_TO_IDX[aa_three_letter]
                                # Use the predicted_prob for model predictions, current_prob for current states
                                if isinstance(aa_data, dict):
                                    pred_prob = aa_data.get('predicted_prob', 0.0)
                                    curr_prob = aa_data.get('current_prob', 0.0)
                                    timestep_predictions[pos_idx, aa_idx] = pred_prob
                                    timestep_states[pos_idx, aa_idx] = curr_prob
                                else:
                                    # Fallback if aa_data is not a dict (shouldn't happen with new format)
                                    timestep_predictions[pos_idx, aa_idx] = float(aa_data)
                                    timestep_states[pos_idx, aa_idx] = float(aa_data)
            
            model_predictions.append(timestep_predictions)
            current_states.append(timestep_states)
        
        # Store predictions and states
        trajectory_analysis_data[f'{pdb_id}_model_predictions'] = model_predictions
        trajectory_analysis_data[f'{pdb_id}_current_states'] = current_states
        
        # Store ground truth
        if 'true_indices' in result and result['true_indices'] is not None:
            ground_truth_onehot = np.zeros((sequence_length, K))
            for pos_idx, true_idx in enumerate(result['true_indices']):
                if 0 <= true_idx < K:
                    ground_truth_onehot[pos_idx, true_idx] = 1.0
            trajectory_analysis_data[f'{pdb_id}_ground_truth'] = ground_truth_onehot
    
    # Save trajectory analysis file
    if trajectory_analysis_data:
        trajectory_filename = f"{timestamp}_{output_prefix}_trajectory_analysis.npz"
        trajectory_filepath = os.path.join(output_dir, trajectory_filename)
        
        np.savez_compressed(trajectory_filepath, **trajectory_analysis_data)
        print(f"Trajectory analysis data saved to: {trajectory_filepath}")
        print(f"Use get_prediction_accuracy('{trajectory_filename}', timestep, pdb_id) to analyze results")
    
    return json_filepath


def sample_multiple_proteins_inpainting_with_trajectory(model, dataset, indices=None, steps=50, T=8.0, t_min=0.0, K=21,
                                                     mask_positions_list=None, known_sequences_list=None, mask_ratio=0.3,
                                                     integration_method='euler', rtol=1e-5, atol=1e-8, max_structures=None,
                                                     output_dir=None, output_prefix=None, args=None):
    """
    Sample multiple proteins with inpainting while tracking trajectories for detailed JSON output.
    
    Args:
        model: Trained model
        dataset: Dataset
        indices: List of structure indices to sample
        steps: Number of sampling steps
        T: Maximum time
        t_min: Minimum time
        K: Number of amino acid classes
        mask_positions_list: List of mask positions for each protein (or None for random masking)
        known_sequences_list: List of known sequences for each protein (or None)
        mask_ratio: Random masking ratio if no specific positions
        integration_method: 'euler' or 'rk45'
        rtol: Relative tolerance for RK45
        atol: Absolute tolerance for RK45
        max_structures: Maximum number of structures to process
        output_dir: Output directory for trajectory analysis (optional)
        output_prefix: Output prefix for trajectory analysis (optional)
        args: Arguments object (optional, for global access)
        
    Returns:
        tuple: (results, structure_names)
    """
    # Import here to avoid circular imports
    from training.inpainting import sample_chain_inpainting_with_trajectory
    
    if indices is None:
        indices = list(range(len(dataset)))
    
    if max_structures is not None:
        indices = indices[:max_structures]
    
    results = []
    structure_names = []
    
    print(f"Sampling {len(indices)} structures with inpainting trajectory tracking...")
    
    for i, idx in enumerate(indices):
        print(f"Processing structure {i+1}/{len(indices)} (index {idx})...")
        
        try:
            data, y_true, mask, time_value, dssp_targets = dataset[idx]  # Unpack 5 values (includes DSSP)
            structure_name = getattr(data, 'name', f'structure_{idx}')
            structure_names.append(structure_name)
            
            # Get structure sequence
            structure_sequence = getattr(data, 'filtered_seq', None)
            if structure_sequence is None:
                raise ValueError(f"No filtered_seq found for structure {idx}")
            
            # Get mask positions and known sequence for this protein
            mask_positions = mask_positions_list[i] if mask_positions_list and i < len(mask_positions_list) else None
            known_sequence = known_sequences_list[i] if known_sequences_list and i < len(known_sequences_list) else None
            
            # Create temporary args object for compatibility
            class TempArgs:
                def __init__(self, dirichlet_concentration, flow_temp=1.0):
                    self.dirichlet_concentration = dirichlet_concentration
                    self.flow_temp = flow_temp
                    self.use_smoothed_targets = getattr(args, 'use_smoothed_targets', False) if args else False
                    self.use_c_factor = getattr(args, 'use_c_factor', False) if args else False
                    
            temp_args = TempArgs(args.dirichlet_concentration if args else 20.0)
            
            # Sample with inpainting trajectory
            if integration_method == 'rk45':
                raise NotImplementedError("RK45 integration was moved to legacy file.")
            else:
                final_probs, pred_seq, mask_tensor, alignment_info, eval_metrics, trajectory_data = sample_chain_inpainting_with_trajectory(
                    model, data, T=T, t_min=t_min, steps=steps, K=K,
                    full_sequence=structure_sequence,  # Use structure sequence as full sequence
                    structure_sequence=structure_sequence,
                    mask_positions=mask_positions,
                    known_sequence=known_sequence,
                    mask_ratio=mask_ratio,
                    args=temp_args
                )
            
            # Calculate legacy accuracy if ground truth available
            accuracy = None
            true_seq = None
            
            # Try to get ground truth from y_true first, then from data.filtered_seq
            if y_true is not None:
                true_seq = y_true.argmax(-1).tolist()
            elif hasattr(data, 'filtered_seq') and data.filtered_seq is not None:
                # Convert filtered_seq to indices using the same mapping as the dataset
                true_seq = []
                for aa_char in data.filtered_seq:
                    if aa_char in SINGLE_TO_TRIPLE:
                        aa3 = SINGLE_TO_TRIPLE[aa_char]
                        if aa3 in AA_TO_IDX:
                            true_seq.append(AA_TO_IDX[aa3])
                        else:
                            true_seq.append(20)  # Unknown
                    else:
                        true_seq.append(20)  # Unknown
            
            # Calculate accuracy if we have ground truth
            if true_seq is not None:
                correct = sum(p == t for p, t in zip(pred_seq, true_seq))
                accuracy = correct / len(pred_seq) * 100

            result = {
                'structure_idx': idx,
                'structure_name': structure_name,
                'length': len(pred_seq),
                'predicted_sequence': pred_seq,
                'true_indices': true_seq,
                'accuracy': accuracy,
                'trajectory_data': trajectory_data,
                'eval_metrics': eval_metrics,  # Detailed evaluation metrics
                'inpainting_mask': mask_tensor.cpu().numpy(),
                'alignment_info': alignment_info,
                'mask_positions': mask_positions,
                'known_sequence': known_sequence,
                'mask_ratio': mask_ratio
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing structure {idx}: {e}")
            structure_names.append(f'structure_{idx}')
            results.append({
                'structure_idx': idx,
                'error': str(e)
            })
    
    # Generate trajectory analysis files if output directory is specified
    if output_dir is not None:
        # Use default output_prefix if not provided
        output_prefix_final = output_prefix if output_prefix is not None else "inpainting_trajectory"
        print(f"\nGenerating inpainting trajectory analysis files for {len(indices)} proteins...")
        try:
            from training.inpainting import generate_inpainting_trajectory_json
            json_filepath = generate_inpainting_trajectory_json(results, structure_names, output_dir, output_prefix_final)
            print(f"Inpainting trajectory JSON saved to: {json_filepath}")
        except Exception as e:
            print(f"Warning: Could not save trajectory analysis files: {e}")
    
    return results, structure_names


def sample_multiple_proteins_with_trajectory(model, dataset, indices=None, steps=50, T=8.0, t_min=0.0, K=21, 
                                           integration_method='euler', rtol=1e-5, atol=1e-8, max_structures=None,
                                           output_dir=None, output_prefix=None, args=None):
    """
    Sample multiple proteins while tracking trajectories for detailed JSON output.
    
    Args:
        model: Trained model
        dataset: Dataset
        indices: List of structure indices to sample
        steps: Number of sampling steps
        T: Maximum time
        K: Number of amino acid classes
        integration_method: 'euler' or 'rk45'
        rtol: Relative tolerance for RK45
        atol: Absolute tolerance for RK45
        max_structures: Maximum number of structures to process
        output_dir: Output directory for trajectory analysis (optional)
        output_prefix: Output prefix for trajectory analysis (optional)
        args: Arguments object (optional, for global access)
        
    Returns:
        tuple: (results, structure_names)
    """
    # Import here to avoid circular imports
    from .sample import sample_chain_with_trajectory
    
    if indices is None:
        indices = list(range(len(dataset)))
    
    if max_structures is not None:
        indices = indices[:max_structures]
    
    results = []
    structure_names = []
    
    print(f"Sampling {len(indices)} structures with trajectory tracking...")
    
    for i, idx in enumerate(indices):
        print(f"Processing structure {i+1}/{len(indices)} (index {idx})...")
        
        try:
            data, y_true, mask, time_value, dssp_targets = dataset[idx]  # Unpack 5 values (includes DSSP)
            structure_name = getattr(data, 'name', f'structure_{idx}')
            structure_names.append(structure_name)
            
            # Sample with trajectory
            # Create temporary args object for dirichlet_concentration
            class TempArgs:
                def __init__(self, dirichlet_concentration):
                    self.dirichlet_concentration = dirichlet_concentration
                    self.flow_temp = 1.0  # Default temperature
                    
            temp_args = TempArgs(args.dirichlet_concentration if args else 20.0)
            
            if integration_method == 'rk45':
                raise NotImplementedError("RK45 integration was moved to legacy file.")
            else:
                final_probs, pred_seq, trajectory, eval_metrics = sample_chain_with_trajectory(
                    model, data, T=T, t_min=t_min, steps=steps, K=K, args=temp_args
                )
            
            # Calculate legacy accuracy if ground truth available (for backwards compatibility)
            accuracy = None
            true_seq = None
            
            # Try to get ground truth from y_true first, then from data.filtered_seq
            if y_true is not None:
                true_seq = y_true.argmax(-1).tolist()
            elif hasattr(data, 'filtered_seq') and data.filtered_seq is not None:
                # Convert filtered_seq to indices using the same mapping as the dataset
                true_seq = []
                for aa_char in data.filtered_seq:
                    if aa_char in SINGLE_TO_TRIPLE:
                        aa3 = SINGLE_TO_TRIPLE[aa_char]
                        if aa3 in AA_TO_IDX:
                            true_seq.append(AA_TO_IDX[aa3])
                        else:
                            true_seq.append(20)  # Unknown
                    else:
                        true_seq.append(20)  # Unknown
            
            # Calculate accuracy if we have ground truth
            if true_seq is not None:
                correct = sum(p == t for p, t in zip(pred_seq, true_seq))
                accuracy = correct / len(pred_seq) * 100

            result = {
                'structure_idx': idx,
                'structure_name': structure_name,
                'length': len(pred_seq),
                'predicted_sequence': pred_seq,
                'true_indices': true_seq,
                'accuracy': accuracy,
                'trajectory_data': trajectory,
                'eval_metrics': eval_metrics  # Detailed evaluation metrics
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing structure {idx}: {e}")
            structure_names.append(f'structure_{idx}')
            results.append({
                'structure_idx': idx,
                'error': str(e)
            })
    
    # Generate trajectory analysis NPZ file if we have < 4 proteins and output directory is specified
    if len(indices) < 4 and output_dir is not None:
        # Use default output_prefix if not provided
        output_prefix_final = output_prefix if output_prefix is not None else "protein_sampling_trajectory"
        if output_prefix is None:
            print(f"\nAutomatically generating trajectory analysis NPZ file for {len(indices)} proteins (< 4 proteins)...")
        else:
            print(f"\nGenerating trajectory analysis NPZ file for {len(indices)} proteins...")
        try:
            from training.trajectory_saver import save_trajectory_analysis_npz
            npz_filepath = save_trajectory_analysis_npz(results, structure_names, output_dir, output_prefix_final)
            print(f"Trajectory analysis saved to: {npz_filepath}")
        except Exception as e:
            print(f"Warning: Could not save trajectory analysis NPZ file: {e}")
    
    return results, structure_names





def sample_multiple_proteins(model, dataset, indices=None, steps=50, T=8.0, K=21, save_probabilities=True, 
                            integration_method='euler', rtol=1e-5, atol=1e-8, args=None):
    """
    Sample sequences for multiple protein structures.
    
    Args:
        model: Trained DFM model
        dataset: CathDataset instance
        indices: List of structure indices to sample (None = all)
        steps: Number of sampling steps
        T: Maximum time
        K: Number of amino acid classes
        save_probabilities: Whether to save probability distributions
        integration_method: 'euler' or 'rk45'
        rtol: Relative tolerance for RK45
        atol: Absolute tolerance for RK45
        args: Arguments object (optional)
        
    Returns:
        List of result dictionaries with evaluation metrics
    """
    # Import here to avoid circular imports
    from .sample import sample_chain
    
    if indices is None:
        indices = list(range(len(dataset)))
    
    # Get number of samples per protein (default to 1 for backward compatibility)
    num_sample_per_protein = getattr(args, 'num_sample_per_protein', 1) if args else 1
    
    results = []
    
    total_samples = len(indices) * num_sample_per_protein
    print(f"Sampling {len(indices)} structures with {num_sample_per_protein} samples per structure ({total_samples} total samples) using {integration_method} integration...")
    
    for i, idx in enumerate(indices):
        print(f"Processing structure {i+1}/{len(indices)} (index {idx}) with {num_sample_per_protein} samples...")
        
        try:
            data, y_true, mask, time_value, dssp_targets = dataset[idx]  # Unpack 5 values (includes DSSP)
            structure_name = getattr(data, 'name', f'structure_{idx}')
            
            # Create temporary args object for dirichlet_concentration if needed
            if args is None:
                class TempArgs:
                    def __init__(self):
                        self.dirichlet_concentration = 20.0
                        self.flow_temp = 1.0
                temp_args = TempArgs()
            else:
                temp_args = args
            
            # Sample multiple sequences for the same structure
            for sample_idx in range(num_sample_per_protein):
                if num_sample_per_protein > 1:
                    print(f"  Sample {sample_idx+1}/{num_sample_per_protein}...")
                
                # Sample sequence with evaluation metrics
                # Each call to sample_chain will use different random noise due to PyTorch's RNG
                if integration_method == 'rk45':
                    raise NotImplementedError("RK45 integration was moved to legacy file.")
                else:
                    final_probs, pred_seq, eval_metrics = sample_chain(
                        model, data, T=T, t_min=0.0, steps=steps, K=K, args=temp_args
                    )
                
                # Calculate legacy accuracy if ground truth available (for backwards compatibility)
                accuracy = None
                true_seq = None
                if y_true is not None:
                    true_seq = y_true.argmax(-1).tolist()
                    correct = sum(p == t for p, t in zip(pred_seq, true_seq))
                    accuracy = correct / len(pred_seq) * 100
                
                # Convert to amino acid names
                predicted_aa = []
                for idx_val in pred_seq:
                    if 0 <= idx_val < len(IDX_TO_AA):
                        predicted_aa.append(IDX_TO_AA[idx_val])
                    else:
                        predicted_aa.append('XXX')
                
                result = {
                    'structure_idx': idx,
                    'structure_name': structure_name,
                    'sample_idx': sample_idx,
                    'length': len(pred_seq),
                    'predicted_indices': pred_seq,
                    'predicted_aa': predicted_aa,
                    'predicted_sequence': ''.join([THREE_TO_ONE[aa] for aa in predicted_aa]),
                    'true_indices': true_seq,
                    'accuracy': accuracy,
                    'eval_metrics': eval_metrics,
                    'final_probabilities': final_probs.cpu().numpy() if save_probabilities else None
                }
                
                # Don't aggregate statistics - each sample will be saved separately
                results.append(result)
            
        except Exception as e:
            print(f"Error processing structure {idx}: {e}")
            results.append({
                'structure_idx': idx,
                'error': str(e)
            })
    
    # Organize results by sample index for separate file saving
    if num_sample_per_protein > 1:
        print(f"\nCompleted sampling: {len(indices)} structures × {num_sample_per_protein} samples = {len(results)} total samples")
        print("Results will be saved in separate files for each sample index.")
    
    return results

def parse_protein_list_from_file(file_path):
    """
    Parse protein names from a text file.
    
    Args:
        file_path: Path to file containing protein names (one per line)
        
    Returns:
        List of protein names
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Protein list file not found: {file_path}")
    
    protein_names = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                protein_names.append(line)
    
    print(f"Loaded {len(protein_names)} protein names from {file_path}")
    return protein_names


def parse_protein_indices_from_string(indices_str):
    """
    Parse protein indices from comma-separated string.
    
    Args:
        indices_str: Comma-separated string of indices (e.g., "0,5,10,15")
        
    Returns:
        List of integers
    """
    try:
        indices = [int(idx.strip()) for idx in indices_str.split(',')]
        print(f"Parsed {len(indices)} protein indices: {indices}")
        return indices
    except ValueError as e:
        raise ValueError(f"Invalid protein indices format: {indices_str}. Expected comma-separated integers.") from e


def parse_protein_names_from_string(names_str):
    """
    Parse protein names from comma-separated string.
    
    Args:
        names_str: Comma-separated string of names (e.g., "1a0o.A,1abc.B,1def.C")
        
    Returns:
        List of protein names
    """
    names = [name.strip() for name in names_str.split(',')]
    print(f"Parsed {len(names)} protein names: {names}")
    return names


def get_indices_for_protein_names(dataset, protein_names, verbose=False):
    """
    Get dataset indices for specific protein names.
    
    Args:
        dataset: CathDataset instance
        protein_names: List of protein names to find
        verbose: Whether to print detailed matching information
        
    Returns:
        Dict mapping protein names to their dataset indices
    """
    if verbose:
        print(f"Looking for {len(protein_names)} proteins in dataset of {len(dataset)} structures...")
    
    name_to_index = {}
    missing_proteins = []
    
    # Create a mapping from protein names to indices
    for idx in range(len(dataset)):
        try:
            data, _, _, _ = dataset[idx]
            protein_name = getattr(data, 'name', None)
            if protein_name:
                name_to_index[protein_name] = idx
                if verbose and protein_name in protein_names:
                    print(f"  Found {protein_name} at index {idx}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load structure at index {idx}: {e}")
            continue
    
    # Find indices for requested proteins
    found_indices = {}
    for protein_name in protein_names:
        if protein_name in name_to_index:
            found_indices[protein_name] = name_to_index[protein_name]
        else:
            missing_proteins.append(protein_name)
    
    if missing_proteins:
        print(f"Warning: Could not find {len(missing_proteins)} proteins in dataset:")
        for protein in missing_proteins:
            print(f"  - {protein}")
        
        # Suggest similar names if available
        available_names = list(name_to_index.keys())
        if available_names:
            print(f"\nAvailable protein names (first 10): {available_names[:10]}")
    
    print(f"Successfully found {len(found_indices)} out of {len(protein_names)} requested proteins")
    return found_indices


def resolve_protein_sampling_mode(args, dataset):
    """
    Resolve which proteins to sample based on command line arguments.
    
    Args:
        args: Parsed command line arguments
        dataset: CathDataset instance
        
    Returns:
        Tuple of (indices_to_sample, sampling_mode_description)
    """
    # Priority order: protein_list > protein_names > protein_indices > protein_name > sample_all > structure_idx
    
    if args.protein_list:
        print(f"Using protein list from file: {args.protein_list}")
        protein_names = parse_protein_list_from_file(args.protein_list)
        name_to_index = get_indices_for_protein_names(dataset, protein_names, verbose=args.verbose)
        indices = list(name_to_index.values())
        return indices, f"protein list from {args.protein_list} ({len(indices)} proteins)"
    
    elif args.protein_names:
        print(f"Using protein names from command line")
        protein_names = parse_protein_names_from_string(args.protein_names)
        name_to_index = get_indices_for_protein_names(dataset, protein_names, verbose=args.verbose)
        indices = list(name_to_index.values())
        return indices, f"protein names from command line ({len(indices)} proteins)"
    
    elif args.protein_indices:
        print(f"Using protein indices from command line")
        indices = parse_protein_indices_from_string(args.protein_indices)
        # Validate indices
        max_idx = len(dataset) - 1
        invalid_indices = [idx for idx in indices if idx < 0 or idx > max_idx]
        if invalid_indices:
            raise ValueError(f"Invalid indices {invalid_indices}. Dataset has indices 0-{max_idx}")
        return indices, f"protein indices from command line ({len(indices)} proteins)"
    
    elif args.protein_name:
        print(f"Using single protein name: {args.protein_name}")
        name_to_index = get_indices_for_protein_names(dataset, [args.protein_name], verbose=args.verbose)
        if not name_to_index:
            raise ValueError(f"Protein '{args.protein_name}' not found in dataset")
        indices = list(name_to_index.values())
        return indices, f"single protein '{args.protein_name}'"
    
    elif args.sample_all:
        if args.max_structures:
            indices = list(range(min(args.max_structures, len(dataset))))
        else:
            indices = list(range(len(dataset)))
        return indices, f"all proteins in {args.split} split ({len(indices)} proteins)"
    
    else:
        # Default to single structure by index
        if args.structure_idx >= len(dataset):
            raise ValueError(f"Structure index {args.structure_idx} out of range (max: {len(dataset)-1})")
        return [args.structure_idx], f"single protein at index {args.structure_idx}"

def save_results_to_files(results, output_prefix, output_dir, model_name=None, split=None, steps=None, T=None):
    """
    Save sampling results to comprehensive output files.
    If multiple samples per protein are present, saves separate files for each sample index.
    
    Args:
        results: List of result dictionaries
        output_prefix: Prefix for output filenames
        output_dir: Output directory
        model_name: Name of the model used
        split: Dataset split used
        steps: Number of sampling steps
        T: Maximum time value
        
    Returns:
        Dictionary with file paths and metadata
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Check if we have multiple samples per protein
    sample_indices = set()
    for result in results:
        if 'sample_idx' in result:
            sample_indices.add(result['sample_idx'])
    
    has_multiple_samples = len(sample_indices) > 1 or (len(sample_indices) == 1 and 0 not in sample_indices)
    
    if has_multiple_samples:
        # Group results by sample index
        results_by_sample = {}
        for result in results:
            sample_idx = result.get('sample_idx', 0)
            if sample_idx not in results_by_sample:
                results_by_sample[sample_idx] = []
            results_by_sample[sample_idx].append(result)
        
        print(f"Multiple samples detected. Saving {len(results_by_sample)} separate sets of files...")
        
        # Save separate files for each sample index
        file_info = {
            'timestamp': timestamp,
            'sample_files': {}
        }
        
        for sample_idx in sorted(results_by_sample.keys()):
            sample_results = results_by_sample[sample_idx]
            sample_suffix = f"_sample{sample_idx+1}"
            
            sample_file_info = _save_single_sample_files(
                sample_results, output_prefix + sample_suffix, output_dir, 
                timestamp, model_name, split, steps, T
            )
            
            file_info['sample_files'][sample_idx] = sample_file_info
            
            print(f"  Sample {sample_idx+1}: {len(sample_results)} structures")
            print(f"    Sequences: {sample_file_info['sequences_file']}")
            print(f"    Probabilities: {sample_file_info['probabilities_file']}")
            print(f"    Metadata: {sample_file_info['metadata_file']}")
        
        return file_info
    
    else:
        # Single sample per protein - use original logic
        return _save_single_sample_files(
            results, output_prefix, output_dir, timestamp, model_name, split, steps, T
        )


def _save_single_sample_files(results, output_prefix, output_dir, timestamp, model_name=None, split=None, steps=None, T=None):
    """
    Save sampling results for a single sample set to comprehensive output files.
    
    Args:
        results: List of result dictionaries for a single sample
        output_prefix: Prefix for output filenames
        output_dir: Output directory  
        timestamp: Timestamp string
        model_name: Name of the model used
        split: Dataset split used
        steps: Number of sampling steps
        T: Maximum time value
        
    Returns:
        Dictionary with file paths and metadata
    """
    # Generate filenames
    #sequences_file = os.path.join(output_dir, f"{timestamp}_{output_prefix}_sequences.csv")
    sequences_file = os.path.join(output_dir, f"{output_prefix}_sequences.csv")
    probabilities_file = os.path.join(output_dir, f"{timestamp}_{output_prefix}_probabilities.npz")
    metadata_file = os.path.join(output_dir, f"{timestamp}_{output_prefix}_metadata.txt")
    
    # 1. Save sequences as CSV
    print(f"  Saving sequences to: {sequences_file}")
    sequences_data = []
    for result in results:
        if 'error' not in result:
            # Convert predicted_sequence properly - check if it's indices or amino acids
            pred_seq = result.get('predicted_sequence', result.get('predicted_indices', []))
            
            # If pred_seq is a list of indices, convert to amino acid string
            if isinstance(pred_seq, list) and pred_seq and isinstance(pred_seq[0], int):
                # It's indices, convert to amino acid string
                aa_string = ''
                for idx in pred_seq:
                    if 0 <= idx < len(IDX_TO_AA):
                        aa_string += THREE_TO_ONE[IDX_TO_AA[idx]]
                    else:
                        aa_string += 'X'
                pred_seq_final = aa_string
            else:
                # It's already a string
                pred_seq_final = str(pred_seq)
            
            # Handle true_sequence properly
            true_seq = result.get('true_sequence', '')
            if result.get('true_indices') and not true_seq:
                # Convert true_indices to amino acid string
                true_indices = result['true_indices']
                if true_indices:
                    true_aa_string = ''
                    for idx in true_indices:
                        if 0 <= idx < len(IDX_TO_AA):
                            true_aa_string += THREE_TO_ONE[IDX_TO_AA[idx]]
                        else:
                            true_aa_string += 'X'
                    true_seq = true_aa_string
            
            sequences_data.append({
                'structure_idx': result['structure_idx'],
                'structure_name': result.get('structure_name', f"structure_{result['structure_idx']}"),
                'length': result['length'],
                'predicted_sequence': pred_seq_final,
                'true_sequence': true_seq,
                'accuracy': result.get('accuracy', None)
            })
    
    if sequences_data:
        df = pd.DataFrame(sequences_data)
        df.to_csv(sequences_file, index=False)
    
    # 2. Save probabilities as NPZ
    successful_results = [r for r in results if 'error' not in r and 'final_probabilities' in r]
    if successful_results:
        print(f"  Saving probabilities to: {probabilities_file}")
        prob_data = {}
        
        # Create amino acid index mapping
        prob_data['aa_index_to_name'] = np.array(IDX_TO_AA)
        
        # Save individual structure probabilities
        for result in successful_results:
            struct_name = result.get('structure_name', f"structure_{result['structure_idx']}")
            prob_data[f'probs_{struct_name}'] = result['final_probabilities']
        
        prob_data['structure_indices'] = np.array([r['structure_idx'] for r in successful_results])
        
        # LEGACY FORMAT: Add support for the old analysis format with struct_{index} keys
        for i, result in enumerate(successful_results):
            struct_idx = result['structure_idx']
            
            # Probabilities for this structure
            prob_data[f'struct_{struct_idx}_probabilities'] = result['final_probabilities']
            
            # True indices (ground truth as indices)
            if 'true_indices' in result and result['true_indices'] is not None:
                prob_data[f'struct_{struct_idx}_true_indices'] = np.array(result['true_indices'])
            elif 'true_sequence' in result and result['true_sequence']:
                # Convert true sequence string to indices
                true_seq = result['true_sequence']
                true_indices = []
                for aa_char in true_seq:
                    # Convert single letter to 3-letter, then to index
                    three_letter = SINGLE_TO_TRIPLE.get(aa_char.upper(), 'XXX')
                    idx = AA_TO_IDX.get(three_letter, 20)  # Default to 'XXX' (index 20)
                    true_indices.append(idx)
                prob_data[f'struct_{struct_idx}_true_indices'] = np.array(true_indices)
            else:
                # Default to all unknown if no ground truth available
                seq_len = result['final_probabilities'].shape[0]
                prob_data[f'struct_{struct_idx}_true_indices'] = np.full(seq_len, 20, dtype=int)  # All 'XXX'
            
            # Predicted indices (argmax of probabilities)
            predicted_indices = np.argmax(result['final_probabilities'], axis=1)
            prob_data[f'struct_{struct_idx}_predicted_indices'] = predicted_indices
        
        np.savez_compressed(probabilities_file, **prob_data)
    else:
        print("  No successful results with probabilities to save")
    
    # 3. Save metadata
    print(f"  Saving metadata to: {metadata_file}")
    with open(metadata_file, 'w') as f:
        f.write("PROTEIN SEQUENCE SAMPLING METADATA\n")
        f.write("="*50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SAMPLING PARAMETERS:\n")
        f.write(f"  Model: {model_name or 'Unknown'}\n")
        f.write(f"  Dataset split: {split or 'Unknown'}\n")
        f.write(f"  Sampling steps: {steps or 'Unknown'}\n")
        f.write(f"  Max time (T): {T or 'Unknown'}\n\n")
        
        f.write("RESULTS SUMMARY:\n")
        total_structures = len(results)
        successful = len([r for r in results if 'error' not in r])
        failed = total_structures - successful
        
        f.write(f"  Total structures: {total_structures}\n")
        f.write(f"  Successful: {successful}\n")
        f.write(f"  Failed: {failed}\n")
        
        if successful > 0:
            accuracies = [r.get('accuracy') for r in results if 'error' not in r and r.get('accuracy') is not None]
            if accuracies:
                avg_acc = np.mean(accuracies)
                f.write(f"  Average accuracy: {avg_acc:.2f}%\n")
                std_acc = np.std(accuracies)
                f.write(f"  STD accuracy: {std_acc:.2f}%\n")
                max_acc = np.max(accuracies)
                f.write(f"  Max accuracy: {max_acc:.2f}%\n")
                min_acc = np.min(accuracies)
                f.write(f"  Min accuracy: {min_acc:.2f}%\n")

    return {
        'sequences_file': sequences_file,
        'probabilities_file': probabilities_file,
        'metadata_file': metadata_file,
        'timestamp': timestamp
    }


def get_prediction_accuracy(file_name, timestep, pdb_id, return_pred=False, check='pred', output_dir='../output/prediction/'):
    """
    Analyze prediction accuracy from trajectory analysis files.
    
    Args:
        file_name: Name of the .npz file containing trajectory analysis
        timestep: Which timestep to analyze 
        pdb_id: ID of the protein structure
        return_pred: Whether to return the predictions
        check: 'pred' for model predictions or 'state' for current states
        output_dir: Directory containing the prediction files
        
    Returns:
        accuracy, cross_entropy_loss[, predictions]
    """
    import numpy as np
    import os
    
    file_path = os.path.join(output_dir, file_name)
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    
    probabilities = np.load(file_path, allow_pickle=True)
    
    if check == 'pred':
        key = f'{pdb_id}_model_predictions'
    elif check == 'state':
        key = f'{pdb_id}_current_states'
    else:
        raise ValueError("check must be either 'pred' or 'state'")
    
    try:
        model_predictions = probabilities[key][timestep]
    except KeyError:
        raise ValueError(f"Index {pdb_id} not found in the probabilities file. Available keys: {list(probabilities.keys())}")
    
    ground_truth = probabilities[f'{pdb_id}_ground_truth']
    
    # Calculate accuracy and categorical cross entropy
    # For accuracy, get the max index for each row in model_predictions
    model_predictions_index = np.argmax(model_predictions, axis=1)
    ground_truth_index = np.argmax(ground_truth, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(model_predictions_index) == np.array(ground_truth_index))
    
    # Manually calculate categorical cross entropy
    ce_loss = -(ground_truth * np.log(model_predictions + 1e-10)).mean(axis=1)
    
    if return_pred:
        return accuracy, ce_loss.mean(), model_predictions
    
    return accuracy, ce_loss.mean()


def parse_protein_list_from_file(file_path):
    """
    Parse protein names from a text file.
    
    Args:
        file_path: Path to file containing protein names (one per line)
        
    Returns:
        List of protein names
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Protein list file not found: {file_path}")
    
    protein_names = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                protein_names.append(line)
    
    print(f"Loaded {len(protein_names)} protein names from {file_path}")
    return protein_names


def parse_protein_indices_from_string(indices_str):
    """
    Parse protein indices from comma-separated string.
    
    Args:
        indices_str: Comma-separated string of indices (e.g., "0,5,10,15")
        
    Returns:
        List of integers
    """
    try:
        indices = [int(idx.strip()) for idx in indices_str.split(',')]
        print(f"Parsed {len(indices)} protein indices: {indices}")
        return indices
    except ValueError as e:
        raise ValueError(f"Invalid protein indices format: {indices_str}. Expected comma-separated integers.") from e


def parse_protein_names_from_string(names_str):
    """
    Parse protein names from comma-separated string.
    
    Args:
        names_str: Comma-separated string of names (e.g., "1a0o.A,1abc.B,1def.C")
        
    Returns:
        List of protein names
    """
    names = [name.strip() for name in names_str.split(',')]
    print(f"Parsed {len(names)} protein names: {names}")
    return names


def get_indices_for_protein_names(dataset, protein_names, verbose=False):
    """
    Get dataset indices for specific protein names.
    
    Args:
        dataset: CathDataset instance
        protein_names: List of protein names to find
        verbose: Whether to print detailed matching information
        
    Returns:
        Dict mapping protein names to their dataset indices
    """
    if verbose:
        print(f"Looking for {len(protein_names)} proteins in dataset of {len(dataset)} structures...")
    
    name_to_index = {}
    missing_proteins = []
    
    # Create a mapping from protein names to indices
    for idx in range(len(dataset)):
        try:
            data, _, _, _ = dataset[idx]
            protein_name = getattr(data, 'name', None)
            if protein_name:
                name_to_index[protein_name] = idx
                if verbose and protein_name in protein_names:
                    print(f"  Found {protein_name} at index {idx}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load structure at index {idx}: {e}")
            continue
    
    # Find indices for requested proteins
    found_indices = {}
    for protein_name in protein_names:
        if protein_name in name_to_index:
            found_indices[protein_name] = name_to_index[protein_name]
        else:
            missing_proteins.append(protein_name)
    
    if missing_proteins:
        print(f"Warning: Could not find {len(missing_proteins)} proteins in dataset:")
        for protein in missing_proteins:
            print(f"  - {protein}")
        
        # Suggest similar names if available
        available_names = list(name_to_index.keys())
        if available_names:
            print(f"\nAvailable protein names (first 10): {available_names[:10]}")
    
    print(f"Successfully found {len(found_indices)} out of {len(protein_names)} requested proteins")
    return found_indices


def resolve_protein_sampling_mode(args, dataset):
    """
    Resolve which proteins to sample based on command line arguments.
    
    Args:
        args: Parsed command line arguments
        dataset: CathDataset instance
        
    Returns:
        Tuple of (indices_to_sample, sampling_mode_description)
    """
    # Priority order: protein_list > protein_names > protein_indices > protein_name > sample_all > structure_idx
    
    if args.protein_list:
        print(f"Using protein list from file: {args.protein_list}")
        protein_names = parse_protein_list_from_file(args.protein_list)
        name_to_index = get_indices_for_protein_names(dataset, protein_names, verbose=args.verbose)
        indices = list(name_to_index.values())
        return indices, f"protein list from {args.protein_list} ({len(indices)} proteins)"
    
    elif args.protein_names:
        print(f"Using protein names from command line")
        protein_names = parse_protein_names_from_string(args.protein_names)
        name_to_index = get_indices_for_protein_names(dataset, protein_names, verbose=args.verbose)
        indices = list(name_to_index.values())
        return indices, f"protein names from command line ({len(indices)} proteins)"
    
    elif args.protein_indices:
        print(f"Using protein indices from command line")
        indices = parse_protein_indices_from_string(args.protein_indices)
        # Validate indices
        max_idx = len(dataset) - 1
        invalid_indices = [idx for idx in indices if idx < 0 or idx > max_idx]
        if invalid_indices:
            raise ValueError(f"Invalid indices {invalid_indices}. Dataset has indices 0-{max_idx}")
        return indices, f"protein indices from command line ({len(indices)} proteins)"
    
    elif args.protein_name:
        print(f"Using single protein name: {args.protein_name}")
        name_to_index = get_indices_for_protein_names(dataset, [args.protein_name], verbose=args.verbose)
        if not name_to_index:
            raise ValueError(f"Protein '{args.protein_name}' not found in dataset")
        indices = list(name_to_index.values())
        return indices, f"single protein '{args.protein_name}'"
    
    elif args.sample_all:
        if args.max_structures:
            indices = list(range(min(args.max_structures, len(dataset))))
        else:
            indices = list(range(len(dataset)))
        return indices, f"all proteins in {args.split} split ({len(indices)} proteins)"
    
    else:
        # Default to single structure by index
        if args.structure_idx >= len(dataset):
            raise ValueError(f"Structure index {args.structure_idx} out of range (max: {len(dataset)-1})")
        return [args.structure_idx], f"single protein at index {args.structure_idx}"


def create_argument_parser():
    """Create and return the argument parser for protein sequence sampling."""
    parser = argparse.ArgumentParser(description="Generate protein sequences using trained DFM model")
    parser.add_argument('--model_path', type=str, 
                       default='../ckpts/model_316.pt',
                       help="Path to the trained model checkpoint")
    
    # Direct PDB/structure input (bypasses dataset)
    parser.add_argument('--pdb_input', type=str, default=None,
                       help="Direct PDB input: local file path, PDB ID (e.g., '1abc'), or PDB ID with chain (e.g., '1fcd.C'). "
                            "Supports: '/path/to/file.pdb', '/path/to/file.cif', '1abc', '1fcd.C'. "
                            "When used, bypasses dataset loading and directly processes the specified structure.")
    
    parser.add_argument('--split_json', type=str,
                       default='../datasets/cath-4.2/chain_set_splits.json',
                       help="Path to dataset splits")
    parser.add_argument('--map_pkl', type=str,
                       default='../datasets/cath-4.2/chain_set_map_with_b_factors.pkl', 
                       help="Path to dataset mapping")
    parser.add_argument('--structure_idx', type=int, default=0,
                       help="Index of structure to use for sampling (ignored if other protein selection options are used)")
    parser.add_argument('--protein_name', type=str, default=None,
                       help="Name of specific protein to sample (e.g., '1a0o.A'). If provided, overrides --structure_idx")
    parser.add_argument('--protein_list', type=str, default=None,
                       help="Path to a file containing list of protein names (one per line) to sample")
    parser.add_argument('--protein_names', type=str, default=None,
                       help="Comma-separated list of protein names to sample (e.g., '1a0o.A,1abc.B,1def.C')")
    parser.add_argument('--protein_indices', type=str, default=None,
                       help="Comma-separated list of dataset indices to sample (e.g., '0,5,10,15')")
    parser.add_argument('--sample_all', action='store_true',
                       help="Sample sequences for all structures in the dataset split")
    parser.add_argument('--max_structures', type=int, default=None,
                       help="Maximum number of structures to sample (if --sample_all is used)")
    parser.add_argument('--output_prefix', type=str, default=None,
                       help="Prefix for output files (defaults to 'protein_sampling_trajectory' for trajectory analysis when <4 proteins)")
    parser.add_argument('--output_dir', type=str, default='../output/prediction/',
                       help="Directory to save output files - trajectory analysis NPZ automatically generated for <4 proteins (default: ../output/prediction/)")
    parser.add_argument('--save_probabilities', action='store_true', default=True,
                       help="Save raw probability distributions (default: True)")
    parser.add_argument('--no_probabilities', action='store_true',
                       help="Don't save raw probabilities (faster, less disk space)")
    parser.add_argument('--detailed_json', action='store_true',
                       help="Generate detailed JSON output with time-step information (automatically enabled for <4 proteins)")
    parser.add_argument('--split', type=str, default='validation',
                       choices=['train', 'validation', 'test'],
                       help="Which dataset split to use")
    parser.add_argument('--flow_temp', type=float, default=1.0,
                       help="Temperature for flow sampling (default: 1.0)")
    parser.add_argument('--time_as_temperature', action='store_true',
                       help="Use time-dependent temperature: flow_temp = t_max - current_time + 0.1 (starts high, cools down)")
    parser.add_argument('--steps', type=int, default=50,
                       help="Number of sampling steps (only used for Euler integration)")
    parser.add_argument('--T', type=float, default=8.0,
                       help="Maximum time (noise level)")
    parser.add_argument('--t_min', type=float, default=0.0,
                       help="Minimum time (initial noise level, default: 0.0)")

    # Integration method selection
    parser.add_argument('--integration_method', type=str, default='euler',
                       choices=['euler'],
                       help="Integration method: 'euler' for fixed-step Euler (default: euler)")
    parser.add_argument('--rtol', type=float, default=1e-5,
                       help="Relative tolerance for RK45 integration (default: 1e-5)")
    parser.add_argument('--atol', type=float, default=1e-8,
                       help="Absolute tolerance for RK45 integration (default: 1e-8)")
    
    # Sampling parameters
    parser.add_argument('--dirichlet_concentration', type=float, default=20.0,
                       help="Concentration parameter for initial Dirichlet distribution (default: 20.0)")
    parser.add_argument('--num_sample_per_protein', type=int, default=1,
                       help="Number of sequences to sample per protein structure with different noise realizations (default: 1)")
    parser.add_argument('--auto_config', action='store_true',
                       help="Automatically extract model configuration from checkpoint")
    parser.add_argument('--verbose', action='store_true',
                       help="Enable verbose debugging output")
    
    # Graph building parameters - Auto-extracted from model checkpoint when possible
    parser.add_argument('--k_neighbors', type=int, default=None,
                       help="Number of nearest neighbors per node in graph construction (auto-extracted from model if not provided)")
    parser.add_argument('--k_farthest', type=int, default=None,
                       help="Number of farthest neighbors per node in graph construction (auto-extracted from model if not provided)")
    parser.add_argument('--k_random', type=int, default=None,
                       help="Number of random neighbors per node in graph construction (auto-extracted from model if not provided)")
    parser.add_argument('--max_edge_dist', type=float, default=None,
                       help="Maximum distance cutoff (Angstroms) for edge creation. Overrides k_neighbors, k_farthest, k_random. Max 80 neighbors per node. (auto-extracted from model if not provided)")
    parser.add_argument('--num_rbf_3d', type=int, default=None,
                       help="Number of RBF features for 3D distances in graph construction (auto-extracted from model if not provided)")
    parser.add_argument('--num_rbf_seq', type=int, default=None,
                       help="Number of RBF features for sequence distances in graph construction (auto-extracted from model if not provided)")
    parser.add_argument('--use_virtual_node', action='store_true',
                       help="Enable virtual node connectivity (auto-extracted from model if not provided)")
    parser.add_argument('--no_virtual_node', action='store_true',
                       help="Disable virtual node connectivity (overrides model setting)")
    
    # RBF distance range parameters
    parser.add_argument('--rbf_3d_min', type=float, default=None,
                       help="Minimum distance for 3D RBF features (auto-extracted from model if not provided, default: 2.0)")
    parser.add_argument('--rbf_3d_max', type=float, default=None,
                       help="Maximum distance for 3D RBF features (auto-extracted from model if not provided, default: 350.0)")
    parser.add_argument('--rbf_3d_spacing', type=str, default=None,
                       choices=['exponential', 'linear', 'log'],
                       help="Spacing method for 3D RBF features (auto-extracted from model if not provided, default: exponential)")
    
    # Velocity function control parameters
    parser.add_argument('--use_c_factor', action='store_true',
                       help="Enable c_factor calculation in velocity function (default: False, sets c_factor=1.0)")
    parser.add_argument('--use_smoothed_targets', action='store_true', default=False,
                       help="Use smoothed targets in velocity computation (default: False). Automatically set to True if --use_smoothed_labels is present.")

    # Structure noise parameters for sampling
    parser.add_argument('--structure_noise_mag_std', type=float, default=None,
                       help="Standard deviation for Gaussian noise added to atom coordinates during sampling. If not provided, uses checkpoint value. Set to 0.0 to disable noise.")
    parser.add_argument('--time_based_struct_noise', type=str, default=None,
                       choices=['increasing', 'decreasing', 'fixed'],
                       help="Time-based structure noise scaling during sampling: 'increasing', 'decreasing', or 'fixed'. If not provided, uses checkpoint value.")
    parser.add_argument('--uncertainty_struct_noise_scaling', action='store_true', default=False,
                       help="Scale structure noise based on uncertainty: more flexible parts get more noise. If not provided, uses checkpoint value.")
    
    # Ensemble sampling parameters
    parser.add_argument('--ensemble_size', type=int, default=1,
                       help="Number of structurally noised replicas to create (default: 1)")
    parser.add_argument('--ensemble_consensus_strength', type=float, default=0.2,
                       help="State consensus strength: 0=independent, 1=full consensus (default: 0.2)")
    parser.add_argument('--ensemble_method', type=str, default='arithmetic', choices=['arithmetic', 'geometric'],
                       help="Ensemble consensus method: arithmetic=mean in probability space, geometric=mean in log space (default: arithmetic)")
    
    # Missing distributed arguments that orchestrator tries to pass
    parser.add_argument('--distributed', action='store_true',
                       help="Enable distributed sampling")
    parser.add_argument('--batch_size', type=int, default=None,
                       help="Batch size for sampling")
    parser.add_argument('--force_batch', action='store_true',
                       help="Force batched sampling for multiple proteins (skip individual sampling phase)")
    parser.add_argument('--threads_per_gpu', type=int, default=2,
                       help="Number of threads per GPU for distributed sampling")
    
    return parser


def process_sampling_args(args):
    """
    Post-process parsed arguments to handle interdependent logic.
    
    Args:
        args: Parsed arguments from create_argument_parser()
        
    Returns:
        args: Modified arguments with resolved dependencies
    """
    # If use_smoothed_labels is True, automatically set use_smoothed_targets to True
    if getattr(args, 'use_smoothed_labels', False):
        args.use_smoothed_targets = True
        print("Setting use_smoothed_targets=True because --use_smoothed_labels was provided")
    elif not hasattr(args, 'use_smoothed_targets'):
        args.use_smoothed_targets = False
    
    return args


def sample_multiple_proteins_batched(model, dataset, indices=None, steps=50, T=8.0, t_min=0.0, K=21, 
                                   save_probabilities=True, integration_method='euler', batch_size=4, args=None):
    """
    Sample sequences for multiple protein structures using proper batched processing.
    
    This function uses the same exact sampling logic as sample_chain but processes
    multiple proteins simultaneously for improved GPU utilization.
    
    Args:
        model: Trained DFM model
        dataset: CathDataset instance
        indices: List of structure indices to sample (None = all)
        steps: Number of sampling steps
        T: Maximum time
        t_min: Minimum time
        K: Number of amino acid classes
        save_probabilities: Whether to save probability distributions
        integration_method: 'euler' or 'rk45'
        batch_size: Number of proteins to process simultaneously
        args: Arguments object (optional)
        
    Returns:
        List of result dictionaries with evaluation metrics
    """
    from torch.distributions import Dirichlet
    from training.collate import collate_fn
    import torch.nn.functional as F
    
    if indices is None:
        indices = list(range(len(dataset)))
    
    device = next(model.parameters()).device
    model.eval()
    results = []
    
    print(f"Batched sampling of {len(indices)} structures using {integration_method} integration...")
    print(f"Batch size: {batch_size}, Steps: {steps}, T: {T}")
    
    # Process proteins in batches
    for batch_start in range(0, len(indices), batch_size):
        batch_end = min(batch_start + batch_size, len(indices))
        batch_indices = indices[batch_start:batch_end]
        current_batch_size = len(batch_indices)
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(indices) + batch_size - 1)//batch_size}: "
              f"structures {batch_start+1}-{batch_end} (indices {batch_indices})")
        
        try:
            # Load batch of proteins
            batch_data = []
            batch_y_true = []
            batch_info = []
            batch_use_virtual = []
            
            for idx in batch_indices:
                data, y_true, mask, time_value, dssp_targets = dataset[idx]  # Unpack 5 values (includes DSSP)
                structure_name = getattr(data, 'name', f'structure_{idx}')
                use_virtual_node = getattr(data, 'use_virtual_node', False)
                
                # Create dummy targets for collate_fn (same as sample_chain)
                dummy_y = torch.zeros(1, K)
                dummy_mask = torch.ones(1, dtype=torch.bool)
                dummy_time = torch.tensor(0.0)
                
                batch_data.append((data, dummy_y, dummy_mask, dummy_time))
                batch_y_true.append(y_true)
                batch_info.append({'idx': idx, 'name': structure_name})
                batch_use_virtual.append(use_virtual_node)
            
            # Collate batch
            batched_data, y_pad, mask_pad, time_batch = collate_fn(batch_data)
            batched_data = batched_data.to(device)
            
            # Handle virtual nodes and get protein sizes (like sample_chain)
            protein_sizes = []
            protein_starts = []
            node_offset = 0
            
            for i in range(current_batch_size):
                # Find where this protein's nodes are in the batch
                protein_mask = (batched_data.batch == i)
                protein_node_count = protein_mask.sum().item()
                
                # Handle virtual nodes like in sample_chain
                use_virtual_node = batch_use_virtual[i]
                if use_virtual_node:
                    actual_size = protein_node_count - 1  # Exclude virtual node
                else:
                    actual_size = protein_node_count
                
                protein_sizes.append(actual_size)
                protein_starts.append(node_offset)
                node_offset += actual_size
            
            # Initialize sequences using Dirichlet noise (same as sample_chain)
            dirichlet_concentration = args.dirichlet_concentration if args else 1
            dirichlet_dist = Dirichlet(dirichlet_concentration * torch.ones(K, device=device))
            
            # Create initial sequences for all real nodes
            x_parts = []
            for size in protein_sizes:
                x_protein = dirichlet_dist.sample((1, size))  # [1, N_protein, K]
                x_parts.append(x_protein)
            
            # Concatenate to create batch tensor [batch_size, max_N, K] with padding
            max_size = max(protein_sizes)
            x_batch = torch.zeros(current_batch_size, max_size, K, device=device)
            
            for i, x_protein in enumerate(x_parts):
                size = protein_sizes[i]
                x_batch[i, :size, :] = x_protein[0, :size, :]
            
            # Time integration (same as sample_chain)
            times = torch.linspace(t_min, T, steps, device=device)
            dt = (T - t_min) / (steps - 1) if steps > 1 else 0
            
            with torch.no_grad():
                time_steps = tqdm(enumerate(times), total=len(times), desc=f"Batched sampling ({current_batch_size} proteins)")
                for i, t_val in time_steps:
                    if i == len(times) - 1:  # Skip last step
                        break
                        
                    t = torch.full((current_batch_size,), t_val, device=device)
                    
                    # Update progress bar with current step info
                    time_steps.set_postfix({'t': f'{t_val:.3f}', 'batch': f'{current_batch_size}'})
                    
                    # Prepare sequence input exactly like sample_chain
                    # The model expects only real node probabilities with shape [batch_size, max_real_nodes, K]
                    # The model will handle virtual nodes internally
                    
                    # Create padded tensor for real nodes only
                    max_real_size = max(protein_sizes)
                    x_model_input = torch.zeros(current_batch_size, max_real_size, K, device=device)
                    
                    for b in range(current_batch_size):
                        real_size = protein_sizes[b]
                        x_model_input[b, :real_size, :] = x_batch[b, :real_size, :]
                    
                    # Get position predictions from model (same as sample_chain)
                    model_output = model(batched_data, t, x_model_input)
                    
                    # Handle DSSP multitask output - model might return (sequence_logits, dssp_logits) tuple
                    if isinstance(model_output, tuple):
                        position_logits = model_output[0]  # Use only sequence logits for sampling
                    else:
                        position_logits = model_output
                    
                    # Apply temperature scaling (same as sample_chain)
                    # Apply time-dependent temperature if requested
                    if args and getattr(args, 'time_as_temperature', False):
                        # Temperature starts high (at t_min) and decreases as we approach t_max
                        flow_temp = T - t_val + 0.1
                    else:
                        flow_temp = args.flow_temp if args and hasattr(args, 'flow_temp') else 1.0
                    predicted_target = torch.softmax(position_logits / flow_temp, dim=-1)
                    
                    # Handle virtual nodes in predictions (same as sample_chain)
                    predicted_target_real = []
                    pred_offset = 0
                    for b in range(current_batch_size):
                        use_virtual = batch_use_virtual[b]
                        protein_total_nodes = (batched_data.batch == b).sum().item()
                        
                        if use_virtual:
                            # Take only real nodes (exclude last virtual node)
                            real_nodes = protein_total_nodes - 1
                            pred_real = predicted_target[pred_offset:pred_offset + real_nodes]
                        else:
                            pred_real = predicted_target[pred_offset:pred_offset + protein_total_nodes]
                        
                        predicted_target_real.append(pred_real)
                        pred_offset += protein_total_nodes
                    
                    # Reshape predictions back to batch format
                    predicted_batch = torch.zeros_like(x_batch)
                    for b, pred_real in enumerate(predicted_target_real):
                        size = protein_sizes[b]
                        predicted_batch[b, :size, :] = pred_real
                    
                    # Compute velocity using conditional flow (same logic as sample_chain)
                    v_batch = torch.zeros_like(x_batch)
                    
                    for b in range(current_batch_size):
                        size = protein_sizes[b]
                        x_protein = x_batch[b:b+1, :size, :]  # [1, N, K]
                        pred_protein = predicted_batch[b:b+1, :size, :]  # [1, N, K]
                        
                        # Try to use conditional flow if available (same as sample_chain)
                        try:
                            cond_flow = model.cond_flow
                            use_virtual = batch_use_virtual[b]
                            v_protein = cond_flow.velocity(
                                x_protein,
                                pred_protein,
                                t[b:b+1],
                                use_virtual_node=use_virtual,
                                use_smoothed_targets=getattr(args, 'use_smoothed_targets', False),
                                use_c_factor=getattr(args, 'use_c_factor', False)
                            )
                        except:
                            # Fallback to simple velocity if cond_flow not available
                            v_protein = pred_protein - x_protein
                        
                        v_batch[b, :size, :] = v_protein[0, :size, :]
                    
                    # Euler step with simplex projection (same as sample_chain)
                    x_new = x_batch + dt * v_batch
                    
                    # Apply simplex projection to each protein
                    for b in range(current_batch_size):
                        size = protein_sizes[b]
                        x_new[b, :size, :] = simplex_proj(x_new[b:b+1, :size, :])[0]
                    
                    x_batch = x_new
            
            # Extract final results for each protein
            for b in range(current_batch_size):
                size = protein_sizes[b]
                final_probabilities = x_batch[b, :size, :].cpu()  # [N, K]
                predicted_sequence = final_probabilities.argmax(-1).tolist()
                
                # Calculate evaluation metrics
                y_true = batch_y_true[b]
                eval_metrics = {}
                accuracy = None
                true_seq = None
                
                if y_true is not None:
                    true_seq = y_true.argmax(-1).tolist() if y_true.dim() > 1 else y_true.tolist()
                    if len(true_seq) == len(predicted_sequence):
                        correct = sum(p == t for p, t in zip(predicted_sequence, true_seq))
                        accuracy = correct / len(predicted_sequence) * 100
                
                # Convert to amino acid names
                predicted_aa = []
                for idx_val in predicted_sequence:
                    if 0 <= idx_val < len(IDX_TO_AA):
                        predicted_aa.append(IDX_TO_AA[idx_val])
                    else:
                        predicted_aa.append('XXX')
                
                result = {
                    'structure_idx': batch_info[b]['idx'],
                    'structure_name': batch_info[b]['name'],
                    'length': len(predicted_sequence),
                    'predicted_indices': predicted_sequence,
                    'predicted_aa': predicted_aa,
                    'predicted_sequence': ''.join([THREE_TO_ONE[aa] for aa in predicted_aa]),
                    'true_indices': true_seq,
                    'accuracy': accuracy,
                    'eval_metrics': eval_metrics,
                    'final_probabilities': final_probabilities.numpy() if save_probabilities else None
                }
                
                results.append(result)
                
        except Exception as e:
            print(f"Error processing batch {batch_indices}: {e}")
            import traceback
            traceback.print_exc()
            # Add error results for each protein in the failed batch
            for idx in batch_indices:
                results.append({
                    'structure_idx': idx,
                    'error': str(e)
                })
    
    return results


def parse_protein_list_from_file(file_path):
    """
    Parse protein names from a text file.
    
    Args:
        file_path: Path to file containing protein names (one per line)
        
    Returns:
        List of protein names
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Protein list file not found: {file_path}")
    
    protein_names = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                protein_names.append(line)
    
    print(f"Loaded {len(protein_names)} protein names from {file_path}")
    return protein_names


def get_indices_for_protein_names(dataset, protein_names, verbose=False):
    """
    Get dataset indices for specific protein names.
    
    Args:
        dataset: CathDataset instance
        protein_names: List of protein names to find
        verbose: Whether to print detailed matching information
        
    Returns:
        Dict mapping protein names to their dataset indices
    """
    if verbose:
        print(f"Looking for {len(protein_names)} proteins in dataset of {len(dataset)} structures...")
    
    name_to_index = {}
    missing_proteins = []
    
    # Create a mapping from protein names to indices
    for idx in range(len(dataset)):
        try:
            data, _, _, _, _ = dataset[idx]  # Unpack 5 values (includes DSSP)
            protein_name = getattr(data, 'name', None)
            if protein_name:
                name_to_index[protein_name] = idx
                if verbose and protein_name in protein_names:
                    print(f"  Found {protein_name} at index {idx}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load structure at index {idx}: {e}")
            continue
    
    # Find indices for requested proteins
    found_indices = {}
    for protein_name in protein_names:
        if protein_name in name_to_index:
            found_indices[protein_name] = name_to_index[protein_name]
        else:
            missing_proteins.append(protein_name)
    
    if missing_proteins:
        print(f"Warning: Could not find {len(missing_proteins)} proteins in dataset:")
        for protein in missing_proteins:
            print(f"  - {protein}")
        
        # Suggest similar names if available
        available_names = list(name_to_index.keys())
        if available_names:
            print(f"\nAvailable protein names (first 10): {available_names[:10]}")
    
    print(f"Successfully found {len(found_indices)} out of {len(protein_names)} requested proteins")
    return found_indices


def resolve_protein_sampling_mode(args, dataset):
    """
    Resolve which proteins to sample based on command line arguments.
    
    Args:
        args: Parsed command line arguments
        dataset: CathDataset instance
        
    Returns:
        Tuple of (indices_to_sample, sampling_mode_description)
    """
    # Priority order: protein_list > protein_name > sample_all > structure_idx
    
    if args.protein_list:
        print(f"Using protein list from file: {args.protein_list}")
        protein_names = parse_protein_list_from_file(args.protein_list)
        name_to_index = get_indices_for_protein_names(dataset, protein_names, verbose=args.verbose)
        indices = list(name_to_index.values())
        return indices, f"protein list from {args.protein_list} ({len(indices)} proteins)"
    
    elif args.protein_name:
        print(f"Using specific protein: {args.protein_name}")
        name_to_index = get_indices_for_protein_names(dataset, [args.protein_name], verbose=args.verbose)
        if args.protein_name in name_to_index:
            return [name_to_index[args.protein_name]], f"specific protein '{args.protein_name}'"
        else:
            raise ValueError(f"Protein '{args.protein_name}' not found in dataset")
    
    elif args.sample_all:
        print("Using sample_all mode - sampling all structures in the dataset")
        indices = list(range(len(dataset)))
        if args.max_structures:
            indices = indices[:args.max_structures]
            return indices, f"all structures (limited to first {args.max_structures})"
        return indices, f"all structures ({len(indices)} total)"
    
    else:
        # Use structure_idx (default mode)
        print(f"Using structure index: {args.structure_idx}")
        if args.structure_idx >= len(dataset):
            raise ValueError(f"Structure index {args.structure_idx} is out of range (dataset has {len(dataset)} structures)")
        return [args.structure_idx], f"single structure at index {args.structure_idx}"
