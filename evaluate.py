import os
import json
import torch
import numpy as np
import networkx as nx
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from train_gnn import GNN
from explainers import (
    get_local_neighborhood,
    run_gnn_explainer,
    SubgraphMCTS,
    CounterfactualExplainer,
    perturb_graph_topology
)
from extensions import (
    homophily_attribution_index,
    hai_shift,
    xai_guided_defense,
)

# Perturbation rate at which the XAI-guided defense is demonstrated.
DEFENSE_RATE = 0.10

def get_top_edges(edge_index, edge_mask, k=5, allowed_nodes=None, target_idx=None):
    """
    Returns up to k undirected (u, v) explanatory edges, highest mask first.

    allowed_nodes: if given, only edges with BOTH endpoints in this set are
        eligible. For node-level explanation this must be the target's k-hop
        receptive field -- edges outside it cannot influence a k-layer GNN's
        prediction, so they must never appear in the explanation. Without this
        filter, top-k taken over the whole graph can select edges nowhere near
        the target (which makes fidelity and robustness meaningless).
    target_idx: if given, edges incident to the target break ties among equal
        mask values, so dense binary masks (e.g. the MCTS subgraph mask) still
        yield target-centered edges instead of arbitrary ones.
    """
    u_arr = edge_index[0].cpu().numpy()
    v_arr = edge_index[1].cpu().numpy()

    candidates = []
    for idx in range(edge_index.size(1)):
        u, v = int(u_arr[idx]), int(v_arr[idx])
        if allowed_nodes is not None and (u not in allowed_nodes or v not in allowed_nodes):
            continue
        m = float(edge_mask[idx])
        if m <= 0.0:
            continue
        incident = 1 if (target_idx is not None and (u == target_idx or v == target_idx)) else 0
        candidates.append((m, incident, u, v))

    # Highest mask first; among equal masks, prefer target-incident edges.
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)

    top_edges = []
    seen = set()
    for m, incident, u, v in candidates:
        e = (min(u, v), max(u, v))
        if e in seen:  # collapse the two directed copies of an undirected edge
            continue
        seen.add(e)
        top_edges.append(e)
        if len(top_edges) == k:
            break
    return top_edges

def compute_fidelity(model, x, edge_index, target_idx, pred_class, explanation_edges):
    """
    Computes Fidelity-minus and Fidelity-plus for a given explanation.
    explanation_edges: List of (u, v) tuples representing the explanatory edges
    """
    model.eval()
    
    # Extract neighborhood
    subset, sub_edge_index, mapping, sub_x, _ = get_local_neighborhood(
        target_idx, 2, edge_index, x, torch.zeros(x.size(0), dtype=torch.long)
    )
    
    target_sub_idx = mapping.item()
    subset_list = subset.cpu().numpy().tolist()
    
    # 1. Original Probability
    with torch.no_grad():
        out_orig = model(x, edge_index)
        prob_orig = torch.exp(out_orig[target_idx])[pred_class].item()
        
    # 2. Fidelity-minus (remove explanation edges)
    # We filter edge_index to remove explanation_edges
    u_edges = edge_index[0].cpu().numpy()
    v_edges = edge_index[1].cpu().numpy()
    
    rem_edges_mask = []
    for idx in range(len(u_edges)):
        u, v = u_edges[idx], v_edges[idx]
        edge_tuple = (min(u, v), max(u, v))
        if edge_tuple not in explanation_edges:
            rem_edges_mask.append(idx)
            
    with torch.no_grad():
        fid_minus_edge_index = edge_index[:, rem_edges_mask]
        out_minus = model(x, fid_minus_edge_index)
        prob_minus = torch.exp(out_minus[target_idx])[pred_class].item()
        
    # 3. Fidelity-plus (keep ONLY explanation edges and remove all other edges in neighborhood)
    # Get all neighborhood edges
    sub_u = sub_edge_index[0].cpu().numpy()
    sub_v = sub_edge_index[1].cpu().numpy()
    
    # Convert explanation edges to set for quick lookup
    exp_set = set(explanation_edges)
    
    keep_edges_mask = []
    # Identify which global edges represent explanation edges or edges outside the neighborhood
    # We remove all neighborhood edges EXCEPT explanation edges
    for idx in range(len(u_edges)):
        u, v = u_edges[idx], v_edges[idx]
        if u in subset_list and v in subset_list:
            # Inside neighborhood: keep only if it is an explanation edge
            edge_tuple = (min(u, v), max(u, v))
            if edge_tuple in exp_set:
                keep_edges_mask.append(idx)
        else:
            # Outside neighborhood: keep it to maintain context
            keep_edges_mask.append(idx)
            
    with torch.no_grad():
        fid_plus_edge_index = edge_index[:, keep_edges_mask]
        out_plus = model(x, fid_plus_edge_index)
        prob_plus = torch.exp(out_plus[target_idx])[pred_class].item()
        
    fid_minus = prob_orig - prob_minus
    fid_plus = prob_orig - prob_plus
    
    return fid_minus, fid_plus, prob_orig, prob_minus, prob_plus

def jaccard_similarity(edges_a, edges_b):
    set_a = set(edges_a)
    set_b = set(edges_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

def main():
    print("=== GNN XAI Robustness Evaluation Suite ===")
    
    data_dir = os.path.join(os.getcwd(), "data")
    model_path = os.path.join(data_dir, "gcn_cora.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please run train_gnn.py first.")
        return
        
    # Load dataset
    dataset = Planetoid(root=data_dir, name="Cora", transform=T.NormalizeFeatures())
    data = dataset[0]
    
    # Load model
    model = GNN(
        in_channels=dataset.num_features,
        hidden_channels=16,
        out_channels=dataset.num_classes,
        model_type='GCN'
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Select nodes to evaluate: we want correctly classified nodes with high prediction confidence
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.exp(logits)
        preds = logits.argmax(dim=-1)
        
    correct_mask = preds == data.y
    confidence, _ = probs.max(dim=-1)
    
    # Filter nodes with high confidence (> 0.8) and in test set
    eval_candidates = torch.where(correct_mask & data.test_mask & (confidence > 0.85))[0].cpu().numpy()
    print(f"Found {len(eval_candidates)} confident evaluation candidates.")
    
    # Pick top 5 nodes spread across different classes
    selected_nodes = []
    seen_classes = set()
    for node in eval_candidates:
        c = data.y[node].item()
        if c not in seen_classes:
            selected_nodes.append(int(node))
            seen_classes.add(c)
        if len(selected_nodes) == 5:
            break
            
    # If we need more nodes to reach 5
    for node in eval_candidates:
        if len(selected_nodes) >= 5:
            break
        if int(node) not in selected_nodes:
            selected_nodes.append(int(node))
            
    print(f"Selected Nodes for evaluation: {selected_nodes}")
    
    # Initialize explainers
    mcts_explainer = SubgraphMCTS(model, num_hops=2, max_rollouts=25, target_size=5)
    cf_explainer = CounterfactualExplainer(model, max_steps=5)
    
    # Robustness parameters
    perturbation_rates = [0.01, 0.05, 0.10, 0.15]
    perturbation_methods = ['random', 'degree', 'homophily']
    
    results = {
        "dataset": "Cora",
        "num_classes": dataset.num_classes,
        "nodes": []
    }
    
    for node_idx in selected_nodes:
        print(f"\nEvaluating Node {node_idx} (True Class: {data.y[node_idx].item()})")
        pred_class = preds[node_idx].item()
        pred_prob = probs[node_idx][pred_class].item()
        
        # 1. Extract local neighborhood subgraph structure for dashboard visualization
        subset, sub_edge_index, mapping, _, _ = get_local_neighborhood(
            node_idx, 2, data.edge_index, data.x, data.y
        )
        subset_list = subset.cpu().numpy().tolist()
        neighborhood = set(subset_list)  # target's 2-hop receptive field
        
        # Convert local neighborhood to networkx for formatting
        local_nodes = []
        for i, global_id in enumerate(subset_list):
            local_nodes.append({
                "id": global_id,
                "label": int(data.y[global_id].item()),
                "is_target": global_id == node_idx,
                "confidence": float(probs[global_id][preds[global_id]].item()),
                "pred_label": int(preds[global_id].item())
            })
            
        local_edges = []
        u_sub = sub_edge_index[0].cpu().numpy()
        v_sub = sub_edge_index[1].cpu().numpy()
        for i in range(len(u_sub)):
            local_edges.append({
                "source": subset_list[u_sub[i]],
                "target": subset_list[v_sub[i]]
            })
            
        # 2. Run Explainers on clean graph
        print("  Running GNNExplainer...")
        mask_gnn = run_gnn_explainer(model, data.x, data.edge_index, node_idx, dataset.num_classes)
        edges_gnn = get_top_edges(data.edge_index, mask_gnn, k=5, allowed_nodes=neighborhood, target_idx=node_idx)

        print("  Running SubgraphMCTS...")
        mask_mcts = mcts_explainer.explain(data.x, data.edge_index, node_idx, pred_class)
        edges_mcts = get_top_edges(data.edge_index, mask_mcts, k=5, allowed_nodes=neighborhood, target_idx=node_idx)

        print("  Running CounterfactualExplainer...")
        mask_cf = cf_explainer.explain(data.x, data.edge_index, node_idx, pred_class)
        edges_cf = get_top_edges(data.edge_index, mask_cf, k=5, allowed_nodes=neighborhood, target_idx=node_idx)
        
        # 3. Compute Fidelity on clean graph
        fid_m_gnn, fid_p_gnn, p_orig, p_m_gnn, p_p_gnn = compute_fidelity(model, data.x, data.edge_index, node_idx, pred_class, edges_gnn)
        fid_m_mcts, fid_p_mcts, _, p_m_mcts, p_p_mcts = compute_fidelity(model, data.x, data.edge_index, node_idx, pred_class, edges_mcts)
        fid_m_cf, fid_p_cf, _, p_m_cf, p_p_cf = compute_fidelity(model, data.x, data.edge_index, node_idx, pred_class, edges_cf)

        # 3b. Homophily Attribution Index (HAI) on the clean explanations
        hai_clean_gnn = homophily_attribution_index(edges_gnn, data.y)
        hai_clean_mcts = homophily_attribution_index(edges_mcts, data.y)
        hai_clean_cf = homophily_attribution_index(edges_cf, data.y)

        node_record = {
            "node_idx": node_idx,
            "true_class": int(data.y[node_idx].item()),
            "pred_class": pred_class,
            "pred_prob": pred_prob,
            "subgraph": {
                "nodes": local_nodes,
                "edges": local_edges
            },
            "explanations": {
                "gnn_explainer": [[int(u), int(v)] for u, v in edges_gnn],
                "subgraph_mcts": [[int(u), int(v)] for u, v in edges_mcts],
                "counterfactual": [[int(u), int(v)] for u, v in edges_cf]
            },
            "fidelity": {
                "gnn_explainer": {"fid_minus": fid_m_gnn, "fid_plus": fid_p_gnn, "p_minus": p_m_gnn, "p_plus": p_p_gnn},
                "subgraph_mcts": {"fid_minus": fid_m_mcts, "fid_plus": fid_p_mcts, "p_minus": p_m_mcts, "p_plus": p_p_mcts},
                "counterfactual": {"fid_minus": fid_m_cf, "fid_plus": fid_p_cf, "p_minus": p_m_cf, "p_plus": p_p_cf}
            },
            "hai_clean": {
                "gnn_explainer": hai_clean_gnn,
                "subgraph_mcts": hai_clean_mcts,
                "counterfactual": hai_clean_cf
            },
            "robustness": {
                "gnn_explainer": {},
                "subgraph_mcts": {},
                "counterfactual": {}
            },
            "defense": {}
        }

        # 3c. Extension 2 -- XAI-Guided Topological Defense.
        # Attack the neighborhood, then prune low-attribution edges and measure recovery.
        print(f"  Running XAI-guided defense (rate={DEFENSE_RATE})...")
        for method in perturbation_methods:
            attacked_edge_index = perturb_graph_topology(
                data.edge_index, data.y, perturbation_rate=DEFENSE_RATE,
                method=method, target_nodes=subset_list
            )
            dres = xai_guided_defense(
                model, data.x, attacked_edge_index, node_idx,
                pred_class, dataset.num_classes, keep_percentile=50.0
            )
            dres["clean_prob"] = pred_prob
            node_record["defense"][method] = dres
        
        # 4. Evaluate robustness under adversarial topology perturbation
        for method in perturbation_methods:
            node_record["robustness"]["gnn_explainer"][method] = []
            node_record["robustness"]["subgraph_mcts"][method] = []
            node_record["robustness"]["counterfactual"][method] = []
            
            for rate in perturbation_rates:
                print(f"    Perturbing using {method} (rate: {rate})...")
                # Apply perturbation (local perturbations centered around target node's neighborhood)
                perturbed_edge_index = perturb_graph_topology(
                    data.edge_index, data.y, perturbation_rate=rate, method=method, target_nodes=subset_list
                )
                
                # Check prediction stability
                with torch.no_grad():
                    out_pert = model(data.x, perturbed_edge_index)
                    pert_pred = out_pert[node_idx].argmax().item()
                    pert_prob = torch.exp(out_pert[node_idx])[pred_class].item()
                    
                # Perturbed graph changes the receptive field, so recompute it.
                p_subset, _, _, _, _ = get_local_neighborhood(
                    node_idx, 2, perturbed_edge_index, data.x, data.y
                )
                p_neighborhood = set(p_subset.cpu().numpy().tolist())

                # Re-explain on perturbed graph
                p_mask_gnn = run_gnn_explainer(model, data.x, perturbed_edge_index, node_idx, dataset.num_classes)
                p_edges_gnn = get_top_edges(perturbed_edge_index, p_mask_gnn, k=5, allowed_nodes=p_neighborhood, target_idx=node_idx)

                p_mask_mcts = mcts_explainer.explain(data.x, perturbed_edge_index, node_idx, pred_class)
                p_edges_mcts = get_top_edges(perturbed_edge_index, p_mask_mcts, k=5, allowed_nodes=p_neighborhood, target_idx=node_idx)

                p_mask_cf = cf_explainer.explain(data.x, perturbed_edge_index, node_idx, pred_class)
                p_edges_cf = get_top_edges(perturbed_edge_index, p_mask_cf, k=5, allowed_nodes=p_neighborhood, target_idx=node_idx)
                
                # Jaccard similarity with original explanations
                jacc_gnn = jaccard_similarity(edges_gnn, p_edges_gnn)
                jacc_mcts = jaccard_similarity(edges_mcts, p_edges_mcts)
                jacc_cf = jaccard_similarity(edges_cf, p_edges_cf)

                # Homophily Attribution Index on the perturbed explanations + shift vs clean
                hai_p_gnn = homophily_attribution_index(p_edges_gnn, data.y)
                hai_p_mcts = homophily_attribution_index(p_edges_mcts, data.y)
                hai_p_cf = homophily_attribution_index(p_edges_cf, data.y)

                node_record["robustness"]["gnn_explainer"][method].append({
                    "rate": rate, "jaccard": jacc_gnn, "pred_flipped": pert_pred != pred_class,
                    "hai": hai_p_gnn, "hai_shift": hai_shift(hai_clean_gnn, hai_p_gnn)})
                node_record["robustness"]["subgraph_mcts"][method].append({
                    "rate": rate, "jaccard": jacc_mcts, "pred_flipped": pert_pred != pred_class,
                    "hai": hai_p_mcts, "hai_shift": hai_shift(hai_clean_mcts, hai_p_mcts)})
                node_record["robustness"]["counterfactual"][method].append({
                    "rate": rate, "jaccard": jacc_cf, "pred_flipped": pert_pred != pred_class,
                    "hai": hai_p_cf, "hai_shift": hai_shift(hai_clean_cf, hai_p_cf)})
                
        results["nodes"].append(node_record)
        
    # Write to web/runs folder for experiment tracking
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    
    web_dir = os.path.join(os.getcwd(), "web")
    runs_dir = os.path.join(web_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    
    # Save the specific run data
    run_filename = f"{run_id}.json"
    out_path = os.path.join(runs_dir, run_filename)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation finished. Results saved to {out_path}")
    
    # Also save to data.json as the active/default fallback run
    default_path = os.path.join(web_dir, "data.json")
    with open(default_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Update runs_list.json catalog
    catalog_path = os.path.join(runs_dir, "runs_list.json")
    catalog = {"runs": []}
    if os.path.exists(catalog_path):
        try:
            with open(catalog_path, 'r') as f:
                catalog = json.load(f)
        except Exception:
            catalog = {"runs": []}
            
    catalog["runs"].append({
        "id": run_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "Cora",
        "model": "GCN",
        "path": f"runs/{run_filename}",
        "nodes_count": len(selected_nodes)
    })
    
    with open(catalog_path, 'w') as f:
        json.dump(catalog, f, indent=2)
    print(f"Runs catalog updated at {catalog_path}")

if __name__ == "__main__":
    main()
