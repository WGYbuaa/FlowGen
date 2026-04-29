import argparse
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
from torchmetrics.classification import BinaryAUROC
from tqdm import tqdm

from seq_model import NonGraphTransformerModel


CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_nested_list(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_nested_list(item)
        else:
            yield item


def build_vocab_and_mappings(json_data):
    print("Building vocabulary from JSON data...")
    nodes = set()
    for sub_graph in json_data:
        for uc in sub_graph:
            for key in ["BF act", "BF obj", "AF act", "AF obj"]:
                flat_list = list(flatten_nested_list(uc.get(key, [])))
                nodes.update(flat_list)

    sorted_nodes = sorted(list(nodes))
    node_to_idx = {node: i + 3 for i, node in enumerate(sorted_nodes)}
    node_to_idx["<PAD>"] = 0
    node_to_idx["<SOS>"] = 1
    node_to_idx["<EOS>"] = 2
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    vocab_size = len(node_to_idx)
    print(f"Vocabulary built. Size: {vocab_size} (including special tokens)")
    return node_to_idx, idx_to_node, vocab_size


def prepare_subgraph_metadata(all_subgraph_data, json_data):
    print("Indexing subgraphs and collecting basic-flow node ids...")
    for i, subgraph_data in enumerate(tqdm(all_subgraph_data, desc="Indexing")):
        subgraph_data.subgraph_index = i
        subgraph_data.node_ids = torch.arange(subgraph_data.x.size(0))
        all_bf_ids_in_subgraph = set()
        if i < len(json_data):
            for uc in json_data[i]:
                all_bf_ids_in_subgraph.update(flatten_nested_list(uc.get("BF act", [])))
                all_bf_ids_in_subgraph.update(flatten_nested_list(uc.get("BF obj", [])))
        subgraph_data.all_bf_node_indices = torch.tensor(
            [
                idx
                for idx, node_id in enumerate(subgraph_data.node_ids.tolist())
                if node_id in all_bf_ids_in_subgraph
            ],
            dtype=torch.long,
        )


def pre_process_uc(subgraph_data, uc_data, node_to_idx, vocab_size):
    keyword_ids = set(uc_data.get(list(uc_data.keys())[0], []))
    bf_act_ids = set(flatten_nested_list(uc_data.get("BF act", [])))
    bf_obj_ids = set(flatten_nested_list(uc_data.get("BF obj", [])))
    bf_ids = bf_act_ids.union(bf_obj_ids)

    keyword_indices = torch.tensor(
        [i for i, node_id in enumerate(subgraph_data.node_ids.tolist()) if node_id in keyword_ids],
        dtype=torch.long,
    )
    bf_node_indices = torch.tensor(
        [i for i, node_id in enumerate(subgraph_data.node_ids.tolist()) if node_id in bf_ids],
        dtype=torch.long,
    )

    num_bf_nodes_in_uc = len(bf_node_indices)
    if num_bf_nodes_in_uc == 0:
        return None

    uc_bf_global_id_to_local_idx_map = {
        global_id.item(): local_idx for local_idx, global_id in enumerate(bf_node_indices)
    }
    branch_target_full = torch.zeros(num_bf_nodes_in_uc, vocab_size)

    decoder_inputs = []
    decoder_targets = []
    af_contexts = []
    af_start_nodes_for_decoder = []

    bf_steps_acts = uc_data.get("BF act", [])
    bf_steps_objs = uc_data.get("BF obj", [])
    af_steps_acts = uc_data.get("AF act", [])

    max_len = 0
    has_any_af_flow = False

    for bf_step_idx, af_flows in enumerate(af_steps_acts):
        if not af_flows:
            continue

        current_bf_acts = bf_steps_acts[bf_step_idx] if bf_step_idx < len(bf_steps_acts) else []
        current_bf_objs = bf_steps_objs[bf_step_idx] if bf_step_idx < len(bf_steps_objs) else []
        current_bf_step_nodes = set(flatten_nested_list(current_bf_acts)).union(
            set(flatten_nested_list(current_bf_objs))
        )

        for af_flow in af_flows:
            af_sequence = tuple(flatten_nested_list(af_flow))
            if not af_sequence:
                continue

            af_start_node_id = af_sequence[0]
            found_valid_branch_point = False
            valid_bf_nodes_for_context = []

            for bf_node_id in current_bf_step_nodes:
                if bf_node_id not in uc_bf_global_id_to_local_idx_map:
                    continue

                bf_node_local_idx = uc_bf_global_id_to_local_idx_map[bf_node_id]
                af_start_node_vocab_idx = node_to_idx.get(af_start_node_id)
                if af_start_node_vocab_idx is None:
                    continue

                branch_target_full[bf_node_local_idx, af_start_node_vocab_idx] = 1.0
                indices = (subgraph_data.node_ids == bf_node_id).nonzero(as_tuple=False).squeeze()
                if indices.numel() == 0:
                    continue

                bf_node_global_idx = indices.item() if indices.numel() == 1 else indices[0].item()
                valid_bf_nodes_for_context.append(bf_node_global_idx)
                found_valid_branch_point = True

            if not found_valid_branch_point:
                continue

            has_any_af_flow = True
            af_contexts.append(list(set(valid_bf_nodes_for_context)))

            af_indices = (subgraph_data.node_ids == af_start_node_id).nonzero(as_tuple=False).squeeze()
            if af_indices.numel() == 0:
                continue

            af_start_node_global_idx_val = af_indices.item() if af_indices.numel() == 1 else af_indices[0].item()
            af_start_nodes_for_decoder.append(af_start_node_global_idx_val)

            input_seq = [node_to_idx["<SOS>"]] + [node_to_idx.get(node, 0) for node in af_sequence]
            target_seq = [node_to_idx.get(node, 0) for node in af_sequence] + [node_to_idx["<EOS>"]]
            decoder_inputs.append(torch.tensor(input_seq, dtype=torch.long))
            decoder_targets.append(torch.tensor(target_seq, dtype=torch.long))
            max_len = max(max_len, len(target_seq))

    pad_idx = node_to_idx["<PAD>"]
    if decoder_inputs:
        for i in range(len(decoder_targets)):
            len_diff = max_len - len(decoder_inputs[i])
            if len_diff > 0:
                decoder_inputs[i] = torch.cat(
                    [decoder_inputs[i], torch.full((len_diff,), pad_idx, dtype=torch.long)]
                )

            len_diff = max_len - len(decoder_targets[i])
            if len_diff > 0:
                decoder_targets[i] = torch.cat(
                    [decoder_targets[i], torch.full((len_diff,), pad_idx, dtype=torch.long)]
                )

        af_sequences_input_tensor = torch.stack(decoder_inputs)
        af_sequences_target_tensor = torch.stack(decoder_targets)
    else:
        af_sequences_input_tensor = torch.empty((0, max_len if max_len > 0 else 1), dtype=torch.long)
        af_sequences_target_tensor = torch.empty((0, max_len if max_len > 0 else 1), dtype=torch.long)

    af_start_nodes_for_decoder_tensor = torch.tensor(af_start_nodes_for_decoder, dtype=torch.long)
    has_branch = has_any_af_flow
    is_branch_point_target = torch.any(branch_target_full > 0.5, dim=1).int()

    return Data(
        x=subgraph_data.x,
        edge_index=subgraph_data.edge_index,
        edge_attr=subgraph_data.edge_attr,
        edge_type=subgraph_data.edge_type,
        subgraph_index=subgraph_data.subgraph_index,
        node_ids=subgraph_data.node_ids,
        all_bf_node_indices=subgraph_data.all_bf_node_indices,
        keyword_indices=keyword_indices,
        bf_node_indices=bf_node_indices,
        branch_target_full=branch_target_full,
        is_branch_point_target=is_branch_point_target,
        has_branch=has_branch,
        af_sequences_input=af_sequences_input_tensor,
        af_sequences_target=af_sequences_target_tensor,
        af_contexts=af_contexts if af_contexts else [],
        af_start_nodes_for_decoder=af_start_nodes_for_decoder_tensor,
    )


def find_best_threshold(preds, targets):
    best_f1 = -1.0
    best_thresh = 0.5
    if targets.sum() == 0:
        return 0.5, 0.0, 0.0, 0.0

    for threshold in torch.arange(0.01, 1.0, 0.01):
        pred_labels = (preds >= threshold.item()).int()
        tp = ((pred_labels == 1) & (targets == 1)).sum().item()
        fp = ((pred_labels == 1) & (targets == 0)).sum().item()
        fn = ((pred_labels == 0) & (targets == 1)).sum().item()
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = threshold.item()

    precision, recall, best_f1_val = compute_prf_at_threshold(preds, targets, best_thresh)
    return best_thresh, precision, recall, best_f1_val


def compute_prf_at_threshold(preds, targets, threshold):
    pred_labels = (preds >= threshold).int()
    tp = ((pred_labels == 1) & (targets == 1)).sum().item()
    fp = ((pred_labels == 1) & (targets == 0)).sum().item()
    fn = ((pred_labels == 0) & (targets == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def build_decoder_memory(model, all_node_embeds, start_node_idx, context_idx_list, device):
    if start_node_idx >= all_node_embeds.shape[0]:
        return None

    if not context_idx_list:
        return None
    context_indices = torch.tensor(context_idx_list, dtype=torch.long, device=device)
    if context_indices.numel() == 0 or torch.max(context_indices) >= all_node_embeds.shape[0]:
        return None

    condition_token = model.build_condition_token(
        all_node_embeds=all_node_embeds,
        start_node_idx=start_node_idx,
        context_indices=context_indices,
    )
    if condition_token is None:
        return None

    return model.build_decoder_memory(all_node_embeds, condition_token)


def evaluate_branch_prediction(model, dataloader, json_data_full, node_to_idx, vocab_size, device, split_name):
    print(f"\n--- {split_name} Branch Point Prediction ---")
    model.eval()

    all_preds_binary = []
    all_targets_binary = []
    group_results = defaultdict(lambda: {"preds": [], "targets": [], "has_branch": False})

    with torch.no_grad():
        iterator = tqdm(dataloader, desc=f"{split_name} Branch", leave=False)
        for data in iterator:
            data = data.to(device)
            subgraph_idx = int(data.subgraph_index.item())
            if subgraph_idx >= len(json_data_full):
                continue

            try:
                all_node_embeds = model.encode_subgraph(data)
                for uc_data in json_data_full[subgraph_idx]:
                    uc_specifics = pre_process_uc(data.cpu(), uc_data, node_to_idx, vocab_size)
                    if uc_specifics is None:
                        continue

                    bf_indices = uc_specifics.bf_node_indices.to(device)
                    keyword_indices = uc_specifics.keyword_indices.to(device)
                    targets_binary = uc_specifics.is_branch_point_target
                    branch_logits = model.predictor(all_node_embeds, bf_indices, keyword_indices)

                    if branch_logits.numel() == 0:
                        continue

                    preds_binary = torch.max(torch.sigmoid(branch_logits), dim=1).values.cpu()
                    all_preds_binary.append(preds_binary)
                    all_targets_binary.append(targets_binary)
                    group_results[subgraph_idx]["preds"].append(preds_binary)
                    group_results[subgraph_idx]["targets"].append(targets_binary)
                    if uc_specifics.has_branch:
                        group_results[subgraph_idx]["has_branch"] = True
            except Exception as exc:
                print(f"{split_name} branch evaluation skipped subgraph {subgraph_idx}: {exc}")
                continue

    if not all_preds_binary:
        return {
            "threshold": 0.5,
            "micro_p": 0.0,
            "micro_r": 0.0,
            "micro_f1": 0.0,
            "macro_p": 0.0,
            "macro_r": 0.0,
            "macro_f1": 0.0,
            "valid_group_count": 0,
        }

    final_preds_binary = torch.cat(all_preds_binary).to(device)
    final_targets_binary = torch.cat(all_targets_binary).int().to(device)
    best_threshold, micro_p, micro_r, micro_f1 = find_best_threshold(final_preds_binary, final_targets_binary)

    macro_p_values = []
    macro_r_values = []
    macro_f1_values = []
    valid_group_count = 0

    for _, results in group_results.items():
        if not results["has_branch"] or not results["preds"]:
            continue

        group_preds = torch.cat(results["preds"]).to(device)
        group_targets = torch.cat(results["targets"]).int().to(device)
        if group_targets.sum() == 0:
            continue

        group_p, group_r, group_f1 = compute_prf_at_threshold(group_preds, group_targets, best_threshold)
        macro_p_values.append(group_p)
        macro_r_values.append(group_r)
        macro_f1_values.append(group_f1)
        valid_group_count += 1

    macro_p = float(np.mean(macro_p_values)) if macro_p_values else 0.0
    macro_r = float(np.mean(macro_r_values)) if macro_r_values else 0.0
    macro_f1 = float(np.mean(macro_f1_values)) if macro_f1_values else 0.0

    print(
        f"{split_name} (Branch Existence - Micro) ; Best Thresh = {best_threshold:.2f} ; "
        f"P = {micro_p:.4f} R = {micro_r:.4f} F1 = {micro_f1:.4f}"
    )
    print(
        f"{split_name} (Branch Existence - Macro) ; "
        f"P = {macro_p:.4f} R = {macro_r:.4f} F1 = {macro_f1:.4f} "
        f"(on {valid_group_count} grouped UCs)"
    )

    return {
        "threshold": best_threshold,
        "micro_p": micro_p,
        "micro_r": micro_r,
        "micro_f1": micro_f1,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "valid_group_count": valid_group_count,
    }


def evaluate_decoder_performance(model, dataloader, json_data_full, node_to_idx, vocab_size, device, split_name):
    print(f"\n--- {split_name} Conditional AF Sequence Generation ---")
    model.eval()
    pad_idx = node_to_idx["<PAD>"]

    all_step_preds = []
    all_step_targets = []

    with torch.no_grad():
        iterator = tqdm(dataloader, desc=f"{split_name} Decoder", leave=False)
        for data in iterator:
            data = data.to(device)
            subgraph_idx = int(data.subgraph_index.item())
            if subgraph_idx >= len(json_data_full):
                continue

            try:
                all_node_embeds = model.encode_subgraph(data)
                for uc_data in json_data_full[subgraph_idx]:
                    uc_specifics = pre_process_uc(data.cpu(), uc_data, node_to_idx, vocab_size)
                    if (
                        uc_specifics is None
                        or not hasattr(uc_specifics, "af_sequences_target")
                        or uc_specifics.af_sequences_target is None
                        or uc_specifics.af_sequences_target.numel() == 0
                    ):
                        continue

                    af_sequences_input = uc_specifics.af_sequences_input.to(device)
                    af_sequences_target = uc_specifics.af_sequences_target.to(device)
                    af_start_nodes = uc_specifics.af_start_nodes_for_decoder.to(device)

                    for seq_idx in range(len(af_sequences_target)):
                        if seq_idx >= len(af_start_nodes):
                            continue

                        start_node_idx = af_start_nodes[seq_idx].item()
                        context_idx_list = (
                            uc_specifics.af_contexts[seq_idx]
                            if hasattr(uc_specifics, "af_contexts") and len(uc_specifics.af_contexts) > seq_idx
                            else []
                        )
                        decoder_memory = build_decoder_memory(
                            model=model,
                            all_node_embeds=all_node_embeds,
                            start_node_idx=start_node_idx,
                            context_idx_list=context_idx_list,
                            device=device,
                        )
                        if decoder_memory is None:
                            continue

                        target_seq = af_sequences_target[seq_idx]
                        valid_length = int((target_seq != pad_idx).sum().item())
                        if valid_length == 0:
                            continue

                        input_tokens = af_sequences_input[seq_idx][:valid_length].unsqueeze(0)
                        target_seq = af_sequences_target[seq_idx]
                        output_logits = model.decoder(input_tokens, decoder_memory).squeeze(0)

                        for step_idx in range(valid_length):
                            target_token_idx = target_seq[step_idx].item()
                            target_vector = torch.zeros_like(output_logits, device=device)
                            target_vector[step_idx, target_token_idx] = 1.0
                            all_step_preds.append(output_logits[step_idx].view(-1))
                            all_step_targets.append(target_vector[step_idx].view(-1))
            except Exception as exc:
                print(f"{split_name} decoder evaluation skipped subgraph {subgraph_idx}: {exc}")
                continue

    if not all_step_preds:
        return {"threshold": 0.5, "p": 0.0, "r": 0.0, "f1": 0.0, "auc": 0.5}

    final_preds = torch.sigmoid(torch.cat(all_step_preds))
    final_targets = torch.cat(all_step_targets).int()
    best_threshold, precision, recall, f1 = find_best_threshold(final_preds, final_targets)
    auc_metric = BinaryAUROC().to(device)
    auc = auc_metric(final_preds.to(device), final_targets.to(device)).item()

    print(
        f"{split_name} (Decoder Performance) ; Best Thresh = {best_threshold:.2f} ; "
        f"P = {precision:.4f} R = {recall:.4f} F1 = {f1:.4f} AUC = {auc:.4f}"
    )
    return {"threshold": best_threshold, "p": precision, "r": recall, "f1": f1, "auc": auc}


def get_args():
    parser = argparse.ArgumentParser(
        description="Non-Graph Transformer baseline for BPP and AFGen (fixed local context)"
    )
    parser.add_argument("--dataset", type=str, default="pub", choices=["pub", "ncet"])
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp")
    parser.add_argument("--output_dir", type=str, default=str(CURRENT_DIR / "outputs"))
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hard_negative_ratio", type=int, default=50)
    parser.add_argument("--early_stop_epoch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="decoder_f1",
        choices=["decoder_f1", "branch_micro_f1", "sum"],
        help="Metric used for checkpoint selection and early stopping.",
    )
    return parser.parse_args()


def resolve_dataset_paths(data_dir, dataset):
    data_dir = Path(data_dir).resolve()
    if dataset == "pub":
        pt_path = data_dir / "ERNIE_pub_withbp_20.pt"
        json_path = data_dir / "Ernie_pub.json"
    else:
        pt_path = data_dir / "ERNIE_ncet_withbp_20.pt"
        json_path = data_dir / "ERNIE_ncet_map.json"

    if not pt_path.exists():
        raise FileNotFoundError(f"Missing PT file: {pt_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON file: {json_path}")
    return pt_path, json_path


def compute_selection_score(branch_metrics, decoder_metrics, selection_metric):
    if selection_metric == "branch_micro_f1":
        return branch_metrics["micro_f1"]
    if selection_metric == "sum":
        return branch_metrics["micro_f1"] + decoder_metrics["f1"]
    return decoder_metrics["f1"]


def main():
    args = get_args()
    if args.batch_size != 1:
        raise ValueError("This baseline expects batch_size=1 because each batch must contain exactly one subgraph.")
    if args.hidden_dim % args.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads for the Transformer baseline.")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    pt_file_path, json_file_path = resolve_dataset_paths(args.data_dir, args.dataset)
    best_model_path = Path(args.output_dir) / (
        f"best_transformer_{args.dataset}_{args.selection_metric}.pth"
    )

    print(f"--- Dataset: {args.dataset}, Fixed Context: local (1-hop) ---")
    print(f"PT file:   {pt_file_path}")
    print(f"JSON file: {json_file_path}")
    print(f"Output dir: {Path(args.output_dir).resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    all_subgraph_data = torch.load(pt_file_path, map_location="cpu", weights_only=False)
    with open(json_file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    node_to_idx, idx_to_node, vocab_size = build_vocab_and_mappings(json_data)
    prepare_subgraph_metadata(all_subgraph_data, json_data)

    split = 30 if args.dataset == "pub" else 338
    train_data = all_subgraph_data[:split]
    eval_data = all_subgraph_data[split : split + 3]
    test_data = all_subgraph_data[split + 3 :]

    dataset_train = torch_geometric.loader.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    dataset_eval = torch_geometric.loader.DataLoader(
        eval_data, batch_size=args.batch_size, shuffle=False
    )
    dataset_test = torch_geometric.loader.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    total_positives = 0.0
    total_elements = 0
    for subgraph_data in train_data:
        if subgraph_data.subgraph_index >= len(json_data):
            continue
        for uc_data in json_data[subgraph_data.subgraph_index]:
            uc_specifics = pre_process_uc(subgraph_data, uc_data, node_to_idx, vocab_size)
            if uc_specifics is None:
                continue
            total_positives += float(uc_specifics.branch_target_full.sum().item())
            total_elements += int(uc_specifics.branch_target_full.numel())

    total_negatives = max(total_elements - total_positives, 1.0)
    pos_weight = total_negatives / max(total_positives, 1.0)
    pos_weight_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float)
    print(f"Positive class weight: {pos_weight:.4f}")

    model = NonGraphTransformerModel(
        input_dim=512,
        hidden_dim=args.hidden_dim,
        vocab_size=vocab_size,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_branch_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction="none")
    loss_decoder_fn = nn.CrossEntropyLoss(ignore_index=node_to_idx["<PAD>"])

    best_selection_score = -1.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_epoch_loss = 0.0
        processed_subgraphs_count = 0

        train_iterator = tqdm(dataset_train, desc=f"Epoch {epoch + 1}/{args.num_epochs}", leave=False)
        for data in train_iterator:
            data = data.to(device)
            subgraph_idx = int(data.subgraph_index.item())
            if subgraph_idx >= len(json_data):
                continue

            optimizer.zero_grad()
            uc_losses_in_graph = []

            try:
                all_node_embeds = model.encode_subgraph(data)
                uc_specifics_list = []
                for uc_data in json_data[subgraph_idx]:
                    specs = pre_process_uc(data.cpu(), uc_data, node_to_idx, vocab_size)
                    if specs is not None:
                        uc_specifics_list.append(specs)

                if not uc_specifics_list:
                    continue

                num_accumulation_steps = len(uc_specifics_list)
                for uc_idx, uc_specifics in enumerate(uc_specifics_list):
                    branch_logits = model.predictor(
                        all_node_embeds,
                        uc_specifics.bf_node_indices.to(device),
                        uc_specifics.keyword_indices.to(device),
                    )
                    branch_targets = uc_specifics.branch_target_full.to(device)
                    loss_branch = torch.tensor(0.0, device=device)

                    if branch_logits.numel() > 0 and branch_targets.numel() > 0:
                        loss_all = loss_branch_fn(branch_logits, branch_targets)
                        pos_mask = branch_targets > 0.5
                        num_pos = pos_mask.sum().item()

                        if num_pos > 0:
                            neg_mask = ~pos_mask
                            num_neg = neg_mask.sum().item()
                            num_hard_neg = min(int(num_pos * args.hard_negative_ratio), num_neg)
                            if num_hard_neg > 0:
                                hard_neg_losses = torch.topk(loss_all[neg_mask], num_hard_neg).values
                                loss_branch = torch.cat([loss_all[pos_mask], hard_neg_losses]).mean()
                            else:
                                loss_branch = loss_all[pos_mask].mean()
                        else:
                            loss_branch = loss_all.mean()

                    loss_decoder = torch.tensor(0.0, device=device)
                    af_inputs = uc_specifics.af_sequences_input.to(device)
                    if af_inputs.numel() > 0:
                        af_targets = uc_specifics.af_sequences_target.to(device)
                        af_start_nodes = uc_specifics.af_start_nodes_for_decoder.to(device)
                        decoder_logits_list = []
                        decoder_targets_list = []

                        for seq_idx in range(len(af_inputs)):
                            if seq_idx >= len(af_start_nodes):
                                continue

                            start_node_idx = af_start_nodes[seq_idx].item()
                            context_idx_list = (
                                uc_specifics.af_contexts[seq_idx]
                                if hasattr(uc_specifics, "af_contexts") and len(uc_specifics.af_contexts) > seq_idx
                                else []
                            )
                            decoder_memory = build_decoder_memory(
                                model=model,
                                all_node_embeds=all_node_embeds,
                                start_node_idx=start_node_idx,
                                context_idx_list=context_idx_list,
                                device=device,
                            )
                            if decoder_memory is None:
                                continue

                            valid_length = int((af_targets[seq_idx] != node_to_idx["<PAD>"]).sum().item())
                            if valid_length == 0:
                                continue

                            input_tokens = af_inputs[seq_idx][:valid_length].unsqueeze(0)
                            output_logits = model.decoder(input_tokens, decoder_memory).squeeze(0)
                            decoder_logits_list.append(output_logits)
                            decoder_targets_list.append(af_targets[seq_idx][:valid_length])

                        if decoder_logits_list:
                            loss_decoder = loss_decoder_fn(
                                torch.cat(decoder_logits_list, dim=0),
                                torch.cat(decoder_targets_list, dim=0),
                            )

                    joint_loss = torch.tensor(0.0, device=device)
                    if loss_branch > 0:
                        joint_loss += (
                            0.5 * torch.exp(-model.log_var_branch) * loss_branch
                            + 0.5 * model.log_var_branch
                        )
                    if loss_decoder > 0:
                        joint_loss += (
                            0.5 * torch.exp(-model.log_var_decoder) * loss_decoder
                            + 0.5 * model.log_var_decoder
                        )

                    if joint_loss > 0:
                        joint_loss_scaled = joint_loss / num_accumulation_steps
                        retain_graph = uc_idx < num_accumulation_steps - 1
                        joint_loss_scaled.backward(retain_graph=retain_graph)
                        uc_losses_in_graph.append(joint_loss.item())
            except Exception as exc:
                print(f"Training skipped subgraph {subgraph_idx}: {exc}")
                optimizer.zero_grad()
                continue

            if uc_losses_in_graph:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_epoch_loss += sum(uc_losses_in_graph)
                processed_subgraphs_count += 1

        avg_epoch_loss = total_epoch_loss / processed_subgraphs_count if processed_subgraphs_count > 0 else 0.0
        epoch_duration = time.time() - epoch_start_time
        print(
            f"Epoch {epoch + 1}/{args.num_epochs}, Avg Loss: {avg_epoch_loss:.4f} "
            f"(Duration: {epoch_duration:.2f}s)"
        )

        branch_metrics = evaluate_branch_prediction(
            model=model,
            dataloader=dataset_eval,
            json_data_full=json_data,
            node_to_idx=node_to_idx,
            vocab_size=vocab_size,
            device=device,
            split_name="EVAL",
        )
        decoder_metrics = evaluate_decoder_performance(
            model=model,
            dataloader=dataset_eval,
            json_data_full=json_data,
            node_to_idx=node_to_idx,
            vocab_size=vocab_size,
            device=device,
            split_name="EVAL",
        )

        selection_score = compute_selection_score(
            branch_metrics=branch_metrics,
            decoder_metrics=decoder_metrics,
            selection_metric=args.selection_metric,
        )
        print(
            f"Checkpoint selection ({args.selection_metric}) = {selection_score:.4f} ; "
            f"best so far = {best_selection_score:.4f}"
        )

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"** New best checkpoint saved to {best_model_path} at epoch {best_epoch} **")

        if epoch + 1 - best_epoch >= args.early_stop_epoch:
            print(
                f"Early stopping triggered: {args.selection_metric} did not improve for "
                f"{args.early_stop_epoch} epochs."
            )
            break

    total_training_time = time.time() - start_time
    print(f"\n--- Training finished in {total_training_time / 60:.2f} minutes ---")

    if best_model_path.exists():
        print(f"Loading best model from epoch {best_epoch}: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("No best checkpoint found. Testing with the last model state.")

    test_branch_metrics = evaluate_branch_prediction(
        model=model,
        dataloader=dataset_test,
        json_data_full=json_data,
        node_to_idx=node_to_idx,
        vocab_size=vocab_size,
        device=device,
        split_name="TEST",
    )
    test_decoder_metrics = evaluate_decoder_performance(
        model=model,
        dataloader=dataset_test,
        json_data_full=json_data,
        node_to_idx=node_to_idx,
        vocab_size=vocab_size,
        device=device,
        split_name="TEST",
    )

    summary = {
        "dataset": args.dataset,
        "fixed_context": "local",
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_selection_score": best_selection_score,
        "branch_metrics": test_branch_metrics,
        "decoder_metrics": test_decoder_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = Path(args.output_dir) / (
        f"summary_transformer_{args.dataset}_{args.selection_metric}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"\nSaved test summary to {summary_path}")
    print(f"--- Script finished at {datetime.now()} ---")


if __name__ == "__main__":
    main()
