# main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
import json
from datetime import datetime
import os
from RGAT_model import AFGeneratorModel # Uses AttentionPredictor now
import random
from torchmetrics.classification import BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryF1Score
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import time

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
            for key in ['BF act', 'BF obj', 'AF act', 'AF obj']:
                flat_list = list(flatten_nested_list(uc.get(key, [])))
                nodes.update(flat_list)
    
    sorted_nodes = sorted(list(nodes))
    node_to_idx = {node: i+3 for i, node in enumerate(sorted_nodes)}
    node_to_idx['<PAD>'] = 0; node_to_idx['<SOS>'] = 1; node_to_idx['<EOS>'] = 2
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    vocab_size = len(node_to_idx)
    print(f"Vocabulary built. Size: {vocab_size} (including special tokens)")
    return node_to_idx, idx_to_node, vocab_size


def pre_process_uc(subgraph_data, uc_data, node_to_idx, vocab_size):
    keyword_ids = set(uc_data.get(list(uc_data.keys())[0], []))
    bf_act_ids = set(flatten_nested_list(uc_data.get('BF act', []))); bf_obj_ids = set(flatten_nested_list(uc_data.get('BF obj', [])))
    bf_ids = bf_act_ids.union(bf_obj_ids)
    
    keyword_indices = torch.tensor([i for i, node_id in enumerate(subgraph_data.node_ids.tolist()) if node_id in keyword_ids], dtype=torch.long)
    bf_node_indices = torch.tensor([i for i, node_id in enumerate(subgraph_data.node_ids.tolist()) if node_id in bf_ids], dtype=torch.long)
    
    num_bf_nodes_in_uc = len(bf_node_indices)
    if num_bf_nodes_in_uc == 0: return None

    uc_bf_global_id_to_local_idx_map = {global_id.item(): local_idx for local_idx, global_id in enumerate(bf_node_indices)}
    branch_target_full = torch.zeros(num_bf_nodes_in_uc, vocab_size) # Keep on CPU
    
    decoder_inputs, decoder_targets, af_contexts, af_start_nodes_for_decoder = [], [], [], []
    bf_steps_acts, bf_steps_objs, af_steps_acts = uc_data.get('BF act', []), uc_data.get('BF obj', []), uc_data.get('AF act', [])
    
    ground_truth_sequences = defaultdict(list)
    max_len = 0; has_any_af_flow = False
    for bf_step_idx, af_flows in enumerate(af_steps_acts):
        if not af_flows: continue
        current_bf_acts = bf_steps_acts[bf_step_idx] if bf_step_idx < len(bf_steps_acts) else []
        current_bf_objs = bf_steps_objs[bf_step_idx] if bf_step_idx < len(bf_steps_objs) else []
        current_bf_step_nodes = set(flatten_nested_list(current_bf_acts)).union(set(flatten_nested_list(current_bf_objs)))
        for af_flow in af_flows:
            af_sequence = tuple(flatten_nested_list(af_flow))
            if not af_sequence: continue
            af_start_node_id = af_sequence[0]
            found_valid_branch_point, valid_bf_nodes_for_context = False, []
            for bf_node_id in current_bf_step_nodes:
                if bf_node_id in uc_bf_global_id_to_local_idx_map:
                    bf_node_local_idx = uc_bf_global_id_to_local_idx_map[bf_node_id]
                    af_start_node_vocab_idx = node_to_idx.get(af_start_node_id)
                    if af_start_node_vocab_idx is not None:
                        branch_target_full[bf_node_local_idx, af_start_node_vocab_idx] = 1.0
                        indices = (subgraph_data.node_ids == bf_node_id).nonzero(as_tuple=False).squeeze()
                        if indices.numel() == 0: continue
                        bf_node_global_idx = indices.item() if indices.numel() == 1 else indices[0].item()
                        valid_bf_nodes_for_context.append(bf_node_global_idx)
                        found_valid_branch_point = True
            if found_valid_branch_point:
                has_any_af_flow = True
                if valid_bf_nodes_for_context:
                    first_bf_global_id = subgraph_data.node_ids[valid_bf_nodes_for_context[0]].item()
                    if first_bf_global_id in uc_bf_global_id_to_local_idx_map:
                         first_bf_local_idx = uc_bf_global_id_to_local_idx_map[first_bf_global_id]
                         ground_truth_sequences[first_bf_local_idx].append(af_sequence)
                af_indices = (subgraph_data.node_ids == af_start_node_id).nonzero(as_tuple=False).squeeze()
                if af_indices.numel() == 0: continue
                af_start_node_global_idx_val = af_indices.item() if af_indices.numel() == 1 else af_indices[0].item()
                af_contexts.append(valid_bf_nodes_for_context)
                af_start_nodes_for_decoder.append(af_start_node_global_idx_val)
                input_seq = [node_to_idx['<SOS>']] + [node_to_idx.get(node, 0) for node in af_sequence]
                target_seq = [node_to_idx.get(node, 0) for node in af_sequence] + [node_to_idx['<EOS>']]
                decoder_inputs.append(torch.tensor(input_seq, dtype=torch.long)); decoder_targets.append(torch.tensor(target_seq, dtype=torch.long))
                if len(target_seq) > max_len: max_len = len(target_seq)

    pad_idx = node_to_idx['<PAD>']
    if decoder_inputs:
        for i in range(len(decoder_targets)):
            len_diff = max_len - len(decoder_inputs[i]); 
            if len_diff > 0: decoder_inputs[i] = torch.cat([decoder_inputs[i], torch.full((len_diff,), pad_idx, dtype=torch.long)])
            len_diff = max_len - len(decoder_targets[i]); 
            if len_diff > 0: decoder_targets[i] = torch.cat([decoder_targets[i], torch.full((len_diff,), pad_idx, dtype=torch.long)])
        af_sequences_input_tensor = torch.stack(decoder_inputs)
        af_sequences_target_tensor = torch.stack(decoder_targets)
    else:
        af_sequences_input_tensor = torch.empty((0, max_len if max_len > 0 else 1), dtype=torch.long)
        af_sequences_target_tensor = torch.empty((0, max_len if max_len > 0 else 1), dtype=torch.long)

    af_start_nodes_for_decoder_tensor = torch.tensor(af_start_nodes_for_decoder, dtype=torch.long)
    has_branch = has_any_af_flow
    is_branch_point_target = torch.any(branch_target_full > 0.5, dim=1).int()

    return Data(
        x=subgraph_data.x, edge_index=subgraph_data.edge_index, edge_attr=subgraph_data.edge_attr,
        edge_type=subgraph_data.edge_type, subgraph_index=subgraph_data.subgraph_index,
        node_ids=subgraph_data.node_ids, all_bf_node_indices=subgraph_data.all_bf_node_indices,
        keyword_indices=keyword_indices, bf_node_indices=bf_node_indices,
        branch_target_full=branch_target_full, is_branch_point_target=is_branch_point_target, has_branch=has_branch,
        af_sequences_input=af_sequences_input_tensor, af_sequences_target=af_sequences_target_tensor,
        af_contexts=af_contexts if af_contexts else [],
        af_start_nodes_for_decoder=af_start_nodes_for_decoder_tensor,
        ground_truth_sequences=ground_truth_sequences if ground_truth_sequences else defaultdict(list)
    )

def find_best_threshold(preds, targets):
    best_f1 = -1.0; best_thresh = 0.5
    if targets.sum() == 0: return 0.5, 0.0, 0.0, 0.0
    for threshold in torch.arange(0.01, 1.0, 0.01):
        pred_labels = (preds >= threshold.item()).int()
        tp = ((pred_labels == 1) & (targets == 1)).sum().item()
        fp = ((pred_labels == 1) & (targets == 0)).sum().item()
        fn = ((pred_labels == 0) & (targets == 1)).sum().item()
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1; best_thresh = threshold.item()
            
    best_pred_labels = (preds >= best_thresh).int()
    tp = ((best_pred_labels == 1) & (targets == 1)).sum().item()
    fp = ((best_pred_labels == 1) & (targets == 0)).sum().item()
    fn = ((best_pred_labels == 0) & (targets == 1)).sum().item()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    best_f1_val = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return best_thresh, p, r, best_f1_val


def evaluate_decoder_performance(model, dataloader, json_data_full, node_to_idx, vocab_size, device):
    print("  Evaluating conditional decoder performance...")
    model.eval(); sos_idx, eos_idx, pad_idx = node_to_idx['<SOS>'], node_to_idx['<EOS>'], node_to_idx['<PAD>']
    all_step_preds, all_step_targets = [], []
    evaluated_graphs = 0
    with torch.no_grad():
        for data in tqdm(dataloader, desc="  Decoder Eval", leave=False):
            data = data.to(device); subgraph_idx = data.subgraph_index.item()
            if subgraph_idx >= len(json_data_full): continue
            list_of_ucs_in_subgraph = json_data_full[subgraph_idx] 
            try:
                all_node_embeds = model.encoder(data.x, data.edge_index, data.edge_type, data.edge_attr)
                if not hasattr(data, 'all_bf_node_indices') or data.all_bf_node_indices.numel() == 0: continue
                encoder_outputs = all_node_embeds[data.all_bf_node_indices]
                graph_processed = False
                for uc_data in list_of_ucs_in_subgraph:
                    uc_specifics_data = pre_process_uc(data.cpu(), uc_data, node_to_idx, vocab_size)
                    if uc_specifics_data is None or not hasattr(uc_specifics_data, 'af_sequences_target') or uc_specifics_data.af_sequences_target is None or uc_specifics_data.af_sequences_target.numel() == 0: continue
                    af_sequences_target = uc_specifics_data.af_sequences_target.to(device)
                    af_contexts_uc = uc_specifics_data.af_contexts
                    af_start_nodes_uc = uc_specifics_data.af_start_nodes_for_decoder.to(device)
                    for i in range(len(af_sequences_target)):
                        target_seq = af_sequences_target[i]
                        context_bf_node_indices_list = af_contexts_uc[i] if i < len(af_contexts_uc) else []
                        if not context_bf_node_indices_list: continue
                        if i >= len(af_start_nodes_uc): continue 
                        start_node_idx = af_start_nodes_uc[i].item()
                        max_context_idx = max(context_bf_node_indices_list) if context_bf_node_indices_list else -1
                        if max_context_idx >= all_node_embeds.shape[0] or start_node_idx >= all_node_embeds.shape[0]: continue
                        context_bf_node_indices = torch.tensor(context_bf_node_indices_list, dtype=torch.long, device=device)
                        context_embeds = all_node_embeds[context_bf_node_indices]
                        if context_embeds.dim() == 1: branch_context_embed = context_embeds.view(1, -1)
                        elif context_embeds.dim() > 2: branch_context_embed = context_embeds.mean(dim=list(range(context_embeds.dim()-1))).view(1, -1)
                        else: branch_context_embed = context_embeds.mean(dim=0).view(1, -1)
                        af_start_node_embed = all_node_embeds[start_node_idx].view(1, -1)
                        initial_hidden_input = torch.cat([branch_context_embed, af_start_node_embed], dim=1)
                        hidden = torch.relu(model.fc_hidden(initial_hidden_input))
                        input_token = torch.tensor([sos_idx], dtype=torch.long, device=device)
                        for t in range(len(target_seq)):
                            target_token_idx = target_seq[t].item()
                            if target_token_idx == pad_idx: break
                            output_logits, hidden = model.decoder(input_token, hidden, encoder_outputs)
                            target_vector = torch.zeros_like(output_logits, device=device); target_vector[0, target_token_idx] = 1.0
                            all_step_preds.append(output_logits.view(-1)); all_step_targets.append(target_vector.view(-1))
                            input_token = target_seq[t].unsqueeze(0)
                            if target_token_idx == eos_idx: break
                    graph_processed = True
                if graph_processed: evaluated_graphs += 1
            except Exception as e: print(f"\nError during decoder evaluation for subgraph {subgraph_idx}: {e}\n"); continue
    print(f"  Decoder evaluation completed on {evaluated_graphs} graphs.")
    if not all_step_preds: return 0.5, 0.0, 0.0, 0.0, 0.5
    print("  Calculating final decoder metrics..."); final_preds, final_targets = torch.sigmoid(torch.cat(all_step_preds)), torch.cat(all_step_targets).int()
    best_thresh, p, r, f1 = find_best_threshold(final_preds, final_targets)
    auc_metric = BinaryAUROC().to(device); auc = auc_metric(final_preds, final_targets).item()
    print("  Decoder metrics calculated.")
    return best_thresh, p, r, f1, auc


def main():
    start_time = time.time(); print(f"--- Script started at {datetime.now()} ---")
    num_epochs, learning_rate, gamma, early_stop_epoch, lambda_branch, lambda_decoder, batch_size = 2000, 3e-4, 0.84, 10, 1.0, 1.0, 1 # 1e-4 5e-5 1e-5 3e-4
    hidden_dim = 128
    hard_negative_ratio = 50 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {device}")
    pt_file_path, json_file_path, best_model_path = '/root/autodl-tmp/gpt_ncet_withbp_20.pt', '/root/autodl-tmp/gpt_ncet_map.json', "best_model_checkpoint.pth"
    print(f"Loading graph data from {pt_file_path}...")
    all_subgraph_data = torch.load(pt_file_path, map_location='cpu', weights_only=False) 
    print(f"Loaded {len(all_subgraph_data)} subgraphs.")
    print(f"Loading JSON data from {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f: json_data = json.load(f)
    print(f"Loaded JSON data corresponding to {len(json_data)} subgraphs.")
    node_to_idx, idx_to_node, vocab_size = build_vocab_and_mappings(json_data)
    print("Pre-adding node IDs and BF indices to subgraph data...")
    for i, subgraph_data in enumerate(tqdm(all_subgraph_data, desc="Pre-adding indices")):
        subgraph_data.subgraph_index = i
        subgraph_data.node_ids = torch.arange(subgraph_data.x.size(0))
        all_bf_ids_in_subgraph = set()
        if i < len(json_data):
            for uc_in_subgraph in json_data[i]:
                 bf_act_ids = set(flatten_nested_list(uc_in_subgraph.get('BF act', []))); bf_obj_ids = set(flatten_nested_list(uc_in_subgraph.get('BF obj', [])))
                 all_bf_ids_in_subgraph.update(bf_act_ids); all_bf_ids_in_subgraph.update(bf_obj_ids)
            subgraph_data.all_bf_node_indices = torch.tensor([idx for idx, node_id in enumerate(subgraph_data.node_ids.tolist()) if node_id in all_bf_ids_in_subgraph], dtype=torch.long)
        else: subgraph_data.all_bf_node_indices = torch.tensor([], dtype=torch.long)
    print("Finished pre-adding indices.")
    print("Splitting data into train/eval/test sets...")
    train_data = all_subgraph_data[:338]; eval_data = all_subgraph_data[338:370]; test_data = all_subgraph_data[370:]  # pub:30/30:33/33 #ncet:338/338:370/370
    print(f"Train graphs: {len(train_data)}, Eval graphs: {len(eval_data)}, Test graphs: {len(test_data)}")
    dataset_train = torch_geometric.loader.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dataset_eval = torch_geometric.loader.DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    dataset_test = torch_geometric.loader.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print("DataLoaders created.")
    print("Calculating positive class weight for loss function based on training set...")
    pos_weight_start_time = time.time(); total_positives, total_elements = 0, 0
    for subgraph_data in tqdm(train_data, desc="Calculating weight", leave=False):
        current_subgraph_index = subgraph_data.subgraph_index
        if current_subgraph_index >= len(json_data): continue
        list_of_ucs_in_subgraph = json_data[current_subgraph_index]
        for uc_data in list_of_ucs_in_subgraph:
            uc_specifics = pre_process_uc(subgraph_data, uc_data, node_to_idx, vocab_size) 
            if uc_specifics is not None and hasattr(uc_specifics, 'branch_target_full'):
                total_positives += uc_specifics['branch_target_full'].sum() # Use dict access
                total_elements += uc_specifics['branch_target_full'].numel()
    total_negatives = total_elements - total_positives
    pos_weight = total_negatives / total_positives if total_positives > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    print(f"Positive weight calculated: {pos_weight:.2f} (took {time.time() - pos_weight_start_time:.2f}s)")
    print("Initializing model and optimizer...")
    model = AFGeneratorModel(num_edge_types=13, in_channels=512, hidden_dim=hidden_dim, edge_dim=1, gamma=gamma, vocab_size=vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_branch_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction='none') 
    loss_decoder_fn = nn.CrossEntropyLoss(ignore_index=node_to_idx['<PAD>'])
    print("Model and optimizer initialized.")
    
    best_F1_micro_pred, best_epoch = 0.0, 0
    best_F1_decoder, best_epoch_decoder = 0.0, 0 

    print(f"--- Starting training for {num_epochs} epochs ---")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_epoch_loss = 0.0
        processed_subgraphs_count = 0 # Count graphs actually processed in the epoch
        
        train_iterator = tqdm(dataset_train, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        for data in train_iterator:
            data = data.to(device); subgraph_idx = data.subgraph_index
            if subgraph_idx >= len(json_data): continue
            list_of_ucs_in_subgraph = json_data[subgraph_idx] 
            
            optimizer.zero_grad() 
            uc_losses_in_graph = [] 
            processed_uc_in_graph_count = 0 # Count UCs processed in this graph for scaling
            
            try:
                all_node_embeds = model.encoder(data.x, data.edge_index, data.edge_type, data.edge_attr)
                
                # Pre-calculate specs for all UCs in the subgraph first
                uc_specifics_list = []
                for uc_data in list_of_ucs_in_subgraph:
                     specs = pre_process_uc(data.cpu(), uc_data, node_to_idx, vocab_size)
                     if specs is not None:
                         uc_specifics_list.append(specs)
                
                if not uc_specifics_list: continue # Skip if no valid UCs in subgraph
                
                num_accumulation_steps = len(uc_specifics_list) # Scale loss by number of UCs

                for uc_idx, uc_specifics_dict in enumerate(uc_specifics_list): 
                    uc_bf_indices = uc_specifics_dict['bf_node_indices'].to(device)
                    uc_kw_indices = uc_specifics_dict['keyword_indices'].to(device)
                    uc_branch_target = uc_specifics_dict['branch_target_full'].to(device)
                    branch_logits = model.predictor(all_node_embeds, uc_bf_indices, uc_kw_indices)
                    
                    loss_branch = torch.tensor(0.0).to(device)
                    if branch_logits.numel() > 0 and uc_branch_target.numel() > 0:
                        targets = uc_branch_target.float()
                        with torch.no_grad():
                            pos_mask = targets > 0.5; neg_mask = ~pos_mask
                            num_positives = pos_mask.sum().item(); num_negatives = neg_mask.sum().item()
                            num_negatives_to_keep = min(int(num_positives * hard_negative_ratio), num_negatives)
                        if num_positives > 0 and num_negatives_to_keep > 0:
                            loss_all = loss_branch_fn(branch_logits, targets)
                            pos_loss = loss_all[pos_mask]; neg_loss = loss_all[neg_mask]
                            if neg_loss.numel() > 0:
                               k = min(num_negatives_to_keep, neg_loss.numel())
                               hard_neg_indices = torch.topk(neg_loss, k=k).indices
                               hard_neg_loss = neg_loss[hard_neg_indices]
                               loss_branch = torch.cat([pos_loss, hard_neg_loss]).mean()
                            else: loss_branch = pos_loss.mean()
                        elif num_positives > 0:
                            loss_all = loss_branch_fn(branch_logits, targets); loss_branch = loss_all[pos_mask].mean()
                        else:
                              loss_all = loss_branch_fn(branch_logits, targets)
                              loss_branch = loss_all[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0).to(device)
                    
                    loss_decoder = torch.tensor(0.0).to(device)
                    # Check existence and content before moving to device
                    if uc_specifics_dict['af_sequences_input'] is not None and uc_specifics_dict['af_sequences_input'].numel() > 0:
                         uc_af_input = uc_specifics_dict['af_sequences_input'].to(device)
                         uc_af_target = uc_specifics_dict['af_sequences_target'].to(device)
                         uc_af_contexts = uc_specifics_dict['af_contexts']
                         uc_af_starts = uc_specifics_dict['af_start_nodes_for_decoder'].to(device)
                         decoder_outputs = []
                         for i in range(len(uc_af_input)):
                            context_bf_node_indices_list = uc_af_contexts[i]
                            if not context_bf_node_indices_list: continue
                            if i >= len(uc_af_starts): continue 
                            start_node_idx = uc_af_starts[i].item()
                            max_context_idx = max(context_bf_node_indices_list) if context_bf_node_indices_list else -1
                            # Perform bounds check against the current all_node_embeds shape
                            if max_context_idx >= all_node_embeds.shape[0] or start_node_idx >= all_node_embeds.shape[0]: continue 
                            
                            context_bf_node_indices = torch.tensor(context_bf_node_indices_list, dtype=torch.long, device=device)
                            context_embeds = all_node_embeds[context_bf_node_indices]
                            if context_embeds.dim() == 1: branch_context_embed = context_embeds.view(1, -1)
                            elif context_embeds.dim() > 2: branch_context_embed = context_embeds.mean(dim=list(range(context_embeds.dim()-1))).view(1, -1)
                            else: branch_context_embed = context_embeds.mean(dim=0).view(1, -1)
                            af_start_node_embed = all_node_embeds[start_node_idx].view(1, -1)
                            initial_hidden_input = torch.cat([branch_context_embed, af_start_node_embed], dim=1)
                            hidden = torch.relu(model.fc_hidden(initial_hidden_input))
                            if not hasattr(data, 'all_bf_node_indices') or data.all_bf_node_indices.numel() == 0: continue
                            encoder_outputs_dec = all_node_embeds[data.all_bf_node_indices]
                            af_input_seq = uc_af_input[i]
                            seq_decoder_outputs = []
                            for t in range(len(af_input_seq)):
                                input_token = af_input_seq[t].unsqueeze(0)
                                output, hidden = model.decoder(input_token, hidden, encoder_outputs_dec)
                                seq_decoder_outputs.append(output)
                            if seq_decoder_outputs: # Only append if something was generated
                                decoder_outputs.append(torch.cat(seq_decoder_outputs, dim=0))

                         if decoder_outputs:
                            temp_loss = 0.0
                            for j in range(len(decoder_outputs)): temp_loss += loss_decoder_fn(decoder_outputs[j], uc_af_target[j])
                            if len(decoder_outputs) > 0: loss_decoder = temp_loss / len(decoder_outputs)
                    
                    loss = (lambda_branch * loss_branch) + (lambda_decoder * loss_decoder)
                    if not torch.isnan(loss) and not torch.isinf(loss) and loss > 0:
                        loss_scaled = loss / num_accumulation_steps 
                        # Determine if this is the last UC in the current subgraph batch
                        is_last_uc = (uc_idx == num_accumulation_steps - 1)
                        loss_scaled.backward(retain_graph=not is_last_uc) 
                        uc_losses_in_graph.append(loss.item()) # Log unscaled loss
            
            except Exception as e:
                print(f"\nError during training for subgraph {subgraph_idx}: {e}\n")
                optimizer.zero_grad() # Clear potentially corrupted gradients before next graph
                continue # Skip optimizer step for this graph

            if uc_losses_in_graph: # Check if any UCs were processed and gradients computed
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_epoch_loss += sum(uc_losses_in_graph) # Add sum of unscaled losses for the graph
                processed_subgraphs_count += 1 # Count graph for averaging epoch loss
            
        # Calculate average loss per processed subgraph
        avg_epoch_loss = total_epoch_loss / processed_subgraphs_count if processed_subgraphs_count > 0 else 0.0
        epoch_duration = time.time() - epoch_start_time
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f} (Duration: {epoch_duration:.2f}s)')

        eval_start_time = time.time(); model.eval()
        print("\n--- Evaluating Branch Point Prediction ---")
        best_thresh_for_eval = 0.5; current_f1_micro = 0.0
        current_f1_decoder = 0.0 
        with torch.no_grad():
            all_preds_binary_eval, all_targets_binary_eval, uc_eval_results_binary = [], [], defaultdict(lambda: {'preds': [], 'targets': [], 'has_branch': False})
            eval_iterator = tqdm(dataset_eval, desc="  Branch Eval", leave=False)
            for data in eval_iterator:
                data = data.to(device); subgraph_idx = data.subgraph_index
                if subgraph_idx >= len(json_data): continue
                list_of_ucs_in_subgraph = json_data[subgraph_idx] 
                try:
                    all_node_embeds = model.encoder(data.x, data.edge_index, data.edge_type, data.edge_attr)
                    for uc_data in list_of_ucs_in_subgraph:
                        uc_specifics = pre_process_uc(data.cpu(), uc_data, node_to_idx, vocab_size) 
                        if uc_specifics is None: continue
                        uc_bf_indices = uc_specifics['bf_node_indices'].to(device)
                        uc_kw_indices = uc_specifics['keyword_indices'].to(device)
                        uc_branch_target_binary = uc_specifics['is_branch_point_target'] 
                        branch_logits = model.predictor(all_node_embeds, uc_bf_indices, uc_kw_indices) 
                        if branch_logits.numel() > 0:
                            preds_binary = torch.max(torch.sigmoid(branch_logits), dim=1).values.cpu()
                            targets_binary = uc_branch_target_binary
                            all_preds_binary_eval.append(preds_binary); all_targets_binary_eval.append(targets_binary)
                            uc_eval_results_binary[subgraph_idx]['preds'].append(preds_binary); uc_eval_results_binary[subgraph_idx]['targets'].append(targets_binary)
                            if uc_specifics['has_branch']: uc_eval_results_binary[subgraph_idx]['has_branch'] = True
                except Exception as e: print(f"\nError during branch evaluation for subgraph {subgraph_idx}: {e}\n"); continue
            
            if all_preds_binary_eval:
                final_preds_binary, final_targets_binary = torch.cat(all_preds_binary_eval).to(device), torch.cat(all_targets_binary_eval).int().to(device)
                best_thresh_for_eval, p_micro, r_micro, f1_micro = find_best_threshold(final_preds_binary, final_targets_binary)
                current_f1_micro = f1_micro
                print(f'EVAL (Branch Existence - Micro) ; Best Thresh = {best_thresh_for_eval:.2f} ; P = {p_micro:.4f} R = {r_micro:.4f} F1 = {f1_micro:.4f}')
                macro_p, macro_r, macro_f1, valid_uc_count = [], [], [], 0
                precision_metric = BinaryPrecision(threshold=best_thresh_for_eval).to(device)
                recall_metric = BinaryRecall(threshold=best_thresh_for_eval).to(device)
                f1_metric = BinaryF1Score(threshold=best_thresh_for_eval).to(device)
                for sg_index, results in uc_eval_results_binary.items():
                    if results['has_branch'] and results['preds']:
                        sg_preds_cpu = torch.cat(results['preds'])
                        sg_targets_cpu = torch.cat(results['targets']).int()
                        if sg_targets_cpu.sum() > 0:
                            sg_preds_dev = sg_preds_cpu.to(device); sg_targets_dev = sg_targets_cpu.to(device)
                            p_uc = precision_metric(sg_preds_dev, sg_targets_dev).item()
                            r_uc = recall_metric(sg_preds_dev, sg_targets_dev).item()
                            f1_uc = f1_metric(sg_preds_dev, sg_targets_dev).item()
                            if not np.isnan(p_uc): macro_p.append(p_uc)
                            if not np.isnan(r_uc): macro_r.append(r_uc)
                            if not np.isnan(f1_uc): macro_f1.append(f1_uc)
                            valid_uc_count += 1
                if valid_uc_count > 0:
                    avg_macro_p = np.mean(macro_p) if macro_p else 0.0; avg_macro_r = np.mean(macro_r) if macro_r else 0.0; avg_macro_f1 = np.mean(macro_f1) if macro_f1 else 0.0
                    print(f'EVAL (Branch Existence - Macro) ; P = {avg_macro_p:.4f} R = {avg_macro_r:.4f} F1 = {avg_macro_f1:.4f} (on {valid_uc_count} UCs)')
                else: print(f'EVAL (Branch Existence - Macro) ; P = 0.0000 R = 0.0000 F1 = 0.0000')

                if f1_micro > best_F1_micro_pred:
                    best_F1_micro_pred, best_epoch = f1_micro, epoch + 1
                    print(f"** New best F1-micro (Branch Existence) on EVAL: {best_F1_micro_pred:.4f} at epoch {best_epoch} **")
                    torch.save(model.state_dict(), best_model_path)
                    print(f"   -> Model checkpoint saved to {best_model_path}")

        print("\n--- Evaluating Conditional AF Sequence Generation ---")
        best_thresh_dec, p_dec, r_dec, f1_dec, auc_dec = evaluate_decoder_performance(model, dataset_eval, json_data, node_to_idx, vocab_size, device) 
        current_f1_decoder = f1_dec
        print(f'EVAL (Decoder Performance) ; Best Thresh = {best_thresh_dec:.2f} ; P = {p_dec:.4f} R = {r_dec:.4f} F1 = {f1_dec:.4f} AUC = {auc_dec:.4f}')

        # Early stopping based on Decoder F1
        if f1_dec > best_F1_decoder:
            best_F1_decoder = f1_dec
            best_epoch_decoder = epoch + 1 
            print(f"** New best F1 (Decoder Performance) on EVAL: {best_F1_decoder:.4f} at epoch {best_epoch_decoder} **")
        
        eval_duration = time.time() - eval_start_time
        print(f"--- Evaluation finished in {eval_duration:.2f}s ---")

        if epoch + 1 - best_epoch_decoder >= early_stop_epoch:
            print(f"Early stopping triggered: Decoder F1 has not improved for {early_stop_epoch} epochs.")
            break

    total_training_time = time.time() - start_time
    print(f"\n--- Training finished in {total_training_time / 60:.2f} minutes ---")

    # --- FINAL TEST ---
    print("\n--- Running final test on the best model ---")
    if os.path.exists(best_model_path):
        print(f"Loading best model from epoch {best_epoch} based on Branch Existence F1-micro {best_F1_micro_pred:.4f}")
        model.load_state_dict(torch.load(best_model_path))
    else: print("No best model checkpoint found. Testing with the last model.")
    
    print("\n--- Testing Branch Point Existence ---")
    best_thresh_test = 0.5
    model.eval()
    with torch.no_grad():
        all_preds_binary_test, all_targets_binary_test, uc_test_results_binary = [], [], defaultdict(lambda: {'preds': [], 'targets': [], 'has_branch': False})
        for data in tqdm(dataset_test, desc="Test Branch Pred", leave=False):
            data = data.to(device); subgraph_idx = data.subgraph_index
            if subgraph_idx >= len(json_data): continue
            list_of_ucs_in_subgraph = json_data[subgraph_idx] 
            try:
                all_node_embeds = model.encoder(data.x, data.edge_index, data.edge_type, data.edge_attr)
                for uc_data in list_of_ucs_in_subgraph:
                    uc_specifics = pre_process_uc(data.cpu(), uc_data, node_to_idx, vocab_size) 
                    if uc_specifics is None: continue
                    uc_bf_indices = uc_specifics['bf_node_indices'].to(device)
                    uc_kw_indices = uc_specifics['keyword_indices'].to(device)
                    uc_branch_target_binary = uc_specifics['is_branch_point_target'] 
                    branch_logits = model.predictor(all_node_embeds, uc_bf_indices, uc_kw_indices) 
                    if branch_logits.numel() > 0:
                        preds_binary = torch.max(torch.sigmoid(branch_logits), dim=1).values.cpu()
                        targets_binary = uc_branch_target_binary
                        all_preds_binary_test.append(preds_binary); all_targets_binary_test.append(targets_binary)
                        uc_test_results_binary[subgraph_idx]['preds'].append(preds_binary); uc_test_results_binary[subgraph_idx]['targets'].append(targets_binary)
                        if uc_specifics['has_branch']: uc_test_results_binary[subgraph_idx]['has_branch'] = True
            except Exception as e: print(f"\nError during test branch prediction for subgraph {subgraph_idx}: {e}\n"); continue
                
        if all_preds_binary_test:
            final_preds_binary, final_targets_binary = torch.cat(all_preds_binary_test).to(device), torch.cat(all_targets_binary_test).int().to(device)
            best_thresh_test, p_micro, r_micro, f1_micro = find_best_threshold(final_preds_binary, final_targets_binary)
            print(f'TEST Results (Branch Existence - Micro) ; Best Thresh = {best_thresh_test:.2f} ; P = {p_micro:.4f} R = {r_micro:.4f} F1 = {f1_micro:.4f}')
            macro_p, macro_r, macro_f1, valid_uc_count = [], [], [], 0
            precision_metric = BinaryPrecision(threshold=best_thresh_test).to(device)
            recall_metric = BinaryRecall(threshold=best_thresh_test).to(device)
            f1_metric = BinaryF1Score(threshold=best_thresh_test).to(device)
            for sg_index, results in uc_test_results_binary.items():
                if results['has_branch'] and results['preds']:
                    sg_preds_cpu = torch.cat(results['preds'])
                    sg_targets_cpu = torch.cat(results['targets']).int()
                    if sg_targets_cpu.sum() > 0:
                        sg_preds_dev = sg_preds_cpu.to(device); sg_targets_dev = sg_targets_cpu.to(device)
                        p_uc = precision_metric(sg_preds_dev, sg_targets_dev).item(); r_uc = recall_metric(sg_preds_dev, sg_targets_dev).item(); f1_uc = f1_metric(sg_preds_dev, sg_targets_dev).item()
                        if not np.isnan(p_uc): macro_p.append(p_uc)
                        if not np.isnan(r_uc): macro_r.append(r_uc)
                        if not np.isnan(f1_uc): macro_f1.append(f1_uc)
                        valid_uc_count += 1
            if valid_uc_count > 0:
                avg_macro_p = np.mean(macro_p) if macro_p else 0.0; avg_macro_r = np.mean(macro_r) if macro_r else 0.0; avg_macro_f1 = np.mean(macro_f1) if macro_f1 else 0.0
                print(f'TEST Results (Branch Existence - Macro) ; P = {avg_macro_p:.4f} R = {avg_macro_r:.4f} F1 = {avg_macro_f1:.4f} (on {valid_uc_count} UCs)')
            else: print(f'TEST Results (Branch Existence - Macro) ; P = 0.0000 R = 0.0000 F1 = 0.0000')

    print("\n--- Testing Conditional AF Sequence Generation ---")
    best_thresh_dec, p_dec, r_dec, f1_dec, auc_dec = evaluate_decoder_performance(model, dataset_test, json_data, node_to_idx, vocab_size, device) 
    print(f'TEST Results (Decoder Performance) ; Best Thresh = {best_thresh_dec:.2f} ; P = {p_dec:.4f} R = {r_dec:.4f} F1 = {f1_dec:.4f} AUC = {auc_dec:.4f}')

    print(f"\n--- Script finished at {datetime.now()} ---")

if __name__ == '__main__':
    main()