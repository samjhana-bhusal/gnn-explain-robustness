import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

# --- Helper function: Get local neighborhood ---
def get_local_neighborhood(node_idx, num_hops, edge_index, x, y):
    """
    Extracts the k-hop subgraph around a target node.
    Returns:
        subset: Original node indices in the neighborhood
        sub_edge_index: Relabeled edge index for the subgraph
        mapping: The index of the target node in the subgraph
        sub_x: Features of the subgraph nodes
    """
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True
    )
    return subset, sub_edge_index, mapping, x[subset], y[subset]

# --- 1. GNNExplainer Wrapper ---
def run_gnn_explainer(model, x, edge_index, node_idx, num_classes, num_hops=2):
    """
    Runs PyG's GNNExplainer to explain the prediction of a target node.

    GNNExplainer is run on the target's k-hop computation subgraph (its canonical
    usage) rather than the whole graph: optimizing a mask over all |E| edges lets
    the weight diffuse onto edges that cannot influence the target, yielding
    poorly localized, low-fidelity explanations. We then map the subgraph edge
    mask back onto the global edge index so downstream code is unchanged.
    """
    model.eval()

    # Restrict to the target's receptive field.
    subset, sub_edge_index, mapping, edge_mask_hop = k_hop_subgraph(
        node_idx=node_idx, num_hops=num_hops, edge_index=edge_index, relabel_nodes=True
    )
    sub_x = x[subset]
    local_target = int(mapping.item())

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type=None, # Focus purely on topological edges
        edge_mask_type='object',
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs'
        ),
    )
    explanation = explainer(sub_x, sub_edge_index, target_index=local_target)
    sub_mask = explanation.edge_mask.cpu().numpy()

    # Map the subgraph mask back to a full-size global edge mask.
    global_mask = np.zeros(edge_index.size(1))
    subset_list = subset.cpu().numpy().tolist()
    edge_lookup = {}
    gu = edge_index[0].cpu().numpy()
    gv = edge_index[1].cpu().numpy()
    for i in range(edge_index.size(1)):
        edge_lookup[(int(gu[i]), int(gv[i]))] = i

    su = sub_edge_index[0].cpu().numpy()
    sv = sub_edge_index[1].cpu().numpy()
    for j in range(sub_edge_index.size(1)):
        g = (subset_list[su[j]], subset_list[sv[j]])
        gi = edge_lookup.get(g)
        if gi is not None:
            global_mask[gi] = sub_mask[j]

    return global_mask

# --- 2. Custom Subgraph MCTS Explainer (Inspired by SubgraphX) ---
class SubgraphMCTS:
    def __init__(self, model, num_hops=2, c_puct=1.0, max_rollouts=30, target_size=5):
        self.model = model
        self.num_hops = num_hops
        self.c_puct = c_puct
        self.max_rollouts = max_rollouts
        self.target_size = target_size

    def explain(self, x, edge_index, node_idx, pred_class):
        self.model.eval()
        # 1. Get k-hop neighborhood
        subset, sub_edge_index, mapping, sub_x, _ = get_local_neighborhood(
            node_idx, self.num_hops, edge_index, x, torch.zeros(x.size(0), dtype=torch.long)
        )
        
        target_sub_idx = mapping.item()
        num_nodes = subset.size(0)
        
        # We represent states as sets of node indices in the relabeled subgraph
        root_nodes = set(range(num_nodes))
        
        # MCTS structures
        # key: tuple representation of active nodes
        visit_counts = {}
        total_rewards = {}
        
        for rollout in range(self.max_rollouts):
            state = root_nodes.copy()
            path = [tuple(sorted(state))]
            
            # Selection
            while len(state) > self.target_size:
                # Find children (states with one node removed, preserving target node)
                children = []
                for node in state:
                    if node != target_sub_idx:
                        child = state - {node}
                        children.append(child)
                
                if not children:
                    break
                
                # Check if children have been visited
                unvisited = [c for c in children if tuple(sorted(c)) not in visit_counts]
                if unvisited:
                    # Randomly pick an unvisited child
                    state = unvisited[np.random.choice(len(unvisited))]
                    path.append(tuple(sorted(state)))
                    break
                
                # Use UCT selection
                parent_key = tuple(sorted(state))
                parent_n = visit_counts[parent_key]
                
                best_uct = -1
                best_child = None
                for child in children:
                    child_key = tuple(sorted(child))
                    q = total_rewards[child_key] / visit_counts[child_key]
                    u = self.c_puct * np.sqrt(np.log(parent_n) / visit_counts[child_key])
                    uct = q + u
                    if uct > best_uct:
                        best_uct = uct
                        best_child = child
                
                state = best_child
                path.append(tuple(sorted(state)))
            
            # Rollout (Simulation to target_size if not already there)
            while len(state) > self.target_size:
                candidates = [n for n in state if n != target_sub_idx]
                if not candidates:
                    break
                state.remove(np.random.choice(candidates))
                
            # Evaluate current subgraph state
            # To evaluate, we construct an edge mask for the subgraph
            sub_nodes = list(state)
            
            # Run GNN model on the pruned graph
            # We can create a mask of active edges
            u_edges = sub_edge_index[0].cpu().numpy()
            v_edges = sub_edge_index[1].cpu().numpy()
            active_edges = []
            for idx in range(len(u_edges)):
                if u_edges[idx] in state and v_edges[idx] in state:
                    active_edges.append(idx)
                    
            # Predict using model
            with torch.no_grad():
                # We feed the whole feature matrix but only sub_edge_index restricted to active edges
                # For nodes not in 'state', they are effectively disconnected
                masked_edge_index = sub_edge_index[:, active_edges]
                out = self.model(sub_x, masked_edge_index)
                prob = torch.exp(out[target_sub_idx])[pred_class].item()
                
            # Reward: high probability of original class + size penalty
            reward = prob - 0.1 * (len(state) / num_nodes)
            
            # Backpropagation
            for p_state in path:
                visit_counts[p_state] = visit_counts.get(p_state, 0) + 1
                total_rewards[p_state] = total_rewards.get(p_state, 0.0) + reward
                
        # Find best subgraph from visited child states
        best_state = None
        best_q = -100
        for state_key, count in visit_counts.items():
            if count > 0:
                q = total_rewards[state_key] / count
                if q > best_q and len(state_key) <= self.target_size:
                    best_q = q
                    best_state = set(state_key)
                    
        if best_state is None:
            best_state = root_nodes
            
        # Map best active nodes back to original edge mask indices
        # If an edge connects two nodes in best_state, mark it as explanatory
        edge_mask = np.zeros(edge_index.size(1))
        
        # We need to map local subgraph edge indices back to global edge index indices
        # Let's rebuild the mapping
        subset_list = subset.cpu().numpy().tolist()
        node_to_local = {global_id: local_id for local_id, global_id in enumerate(subset_list)}
        
        global_u = edge_index[0].cpu().numpy()
        global_v = edge_index[1].cpu().numpy()
        
        for idx in range(len(global_u)):
            u, v = global_u[idx], global_v[idx]
            if u in node_to_local and v in node_to_local:
                if node_to_local[u] in best_state and node_to_local[v] in best_state:
                    edge_mask[idx] = 1.0
                    
        return edge_mask

# --- 3. Counterfactual Explainer ---
class CounterfactualExplainer:
    def __init__(self, model, max_steps=5):
        self.model = model
        self.max_steps = max_steps
        
    def explain(self, x, edge_index, node_idx, pred_class):
        """
        Finds the minimal set of edge removals in the local neighborhood
        that flips the prediction of the target node.
        """
        self.model.eval()
        
        # 1. Get neighborhood edges
        subset, sub_edge_index, mapping, sub_x, _ = get_local_neighborhood(
            node_idx, 2, edge_index, x, torch.zeros(x.size(0), dtype=torch.long)
        )
        
        target_sub_idx = mapping.item()
        num_edges = sub_edge_index.size(1)
        
        active_edge_indices = list(range(num_edges))
        removed_edges = []
        
        # Greedy deletion search
        for step in range(self.max_steps):
            # Check current prediction
            with torch.no_grad():
                current_edge_index = sub_edge_index[:, active_edge_indices]
                out = self.model(sub_x, current_edge_index)
                current_pred = out[target_sub_idx].argmax().item()
                
            if current_pred != pred_class:
                # Prediction successfully flipped!
                break
                
            best_drop = -1
            edge_to_remove = None
            
            for edge_idx in active_edge_indices:
                # Temporary remove
                temp_indices = [e for e in active_edge_indices if e != edge_idx]
                with torch.no_grad():
                    temp_edge_index = sub_edge_index[:, temp_indices]
                    out = self.model(sub_x, temp_edge_index)
                    probs = torch.exp(out[target_sub_idx])
                    prob_orig = probs[pred_class].item()
                    prob_other = 1.0 - prob_orig
                    
                # We want to minimize the original class probability
                # Large drop is good
                drop = 1.0 - prob_orig
                if drop > best_drop:
                    best_drop = drop
                    edge_to_remove = edge_idx
            
            if edge_to_remove is not None:
                active_edge_indices.remove(edge_to_remove)
                removed_edges.append(edge_to_remove)
            else:
                break
                
        # Generate global edge mask showing removed edges
        edge_mask = np.zeros(edge_index.size(1))
        subset_list = subset.cpu().numpy().tolist()
        
        global_u = edge_index[0].cpu().numpy()
        global_v = edge_index[1].cpu().numpy()
        
        # Map local subgraph edges to global indices
        for local_edge_idx in removed_edges:
            lu = sub_edge_index[0, local_edge_idx].item()
            lv = sub_edge_index[1, local_edge_idx].item()
            gu = subset_list[lu]
            gv = subset_list[lv]
            # Find matching global edge
            matches = np.where((global_u == gu) & (global_v == gv))[0]
            if len(matches) > 0:
                edge_mask[matches[0]] = 1.0
                
        return edge_mask

# --- 4. Adversarial Topology Perturbation Engine ---
def perturb_graph_topology(edge_index, y, perturbation_rate=0.05, method='random', target_nodes=None):
    """
    Applies edge-level perturbations to the graph structure.
    Args:
        edge_index: Original edge index tensor
        y: Node labels
        perturbation_rate: Fraction of edges to perturb (add or delete)
        method: 'random', 'degree', or 'homophily'
        target_nodes: Nodes in whose neighborhood we can add/remove edges
    """
    num_nodes = y.size(0)
    num_edges = edge_index.size(1)
    num_perturb = int(num_edges * perturbation_rate)
    
    if num_perturb == 0:
        return edge_index.clone()
        
    edges_set = set(zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()))
    new_edges = list(edges_set.copy())
    
    if method == 'random':
        # Random edge additions and deletions
        num_adds = num_perturb // 2
        num_dels = num_perturb - num_adds
        
        # Deletes
        for _ in range(num_dels):
            if new_edges:
                idx = np.random.choice(len(new_edges))
                new_edges.pop(idx)
                
        # Additions
        for _ in range(num_adds):
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u != v and (u, v) not in edges_set and (v, u) not in edges_set:
                new_edges.append((u, v))
                
    elif method == 'degree':
        # Target high-degree nodes by adding edges between target node's local neighborhood and high degree nodes
        # Connect low-degree nodes to high-degree nodes to dilute GCN signals
        degrees = np.zeros(num_nodes)
        for u, v in edges_set:
            degrees[u] += 1
            degrees[v] += 1
            
        high_degree_nodes = np.argsort(degrees)[-50:] # Top 50 high degree nodes
        
        # Let's add edges between target_nodes and high degree nodes
        if target_nodes is not None:
            nodes_to_link = list(target_nodes)
            for _ in range(num_perturb):
                if len(nodes_to_link) > 0:
                    u = np.random.choice(nodes_to_link)
                    v = np.random.choice(high_degree_nodes)
                    if u != v and (u, v) not in edges_set:
                        new_edges.append((u, v))
        else:
            for _ in range(num_perturb):
                u = np.random.randint(0, num_nodes)
                v = np.random.choice(high_degree_nodes)
                if u != v and (u, v) not in edges_set:
                    new_edges.append((u, v))
                    
    elif method == 'homophily':
        # Homophily disrupting: add edges between nodes with DIFFERENT labels
        labels = y.cpu().numpy()
        added = 0
        trials = 0
        while added < num_perturb and trials < num_perturb * 10:
            trials += 1
            if target_nodes is not None and len(target_nodes) > 0:
                u = np.random.choice(list(target_nodes))
            else:
                u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u != v and labels[u] != labels[v] and (u, v) not in edges_set:
                new_edges.append((u, v))
                added += 1
                
    # Rebuild edge index tensor
    if len(new_edges) > 0:
        new_edge_index = torch.tensor(list(zip(*new_edges)), dtype=torch.long, device=edge_index.device)
    else:
        new_edge_index = edge_index.clone()
        
    return new_edge_index
