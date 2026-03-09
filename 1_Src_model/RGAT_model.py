# RGAT_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv
import math # Needed for Positional Encoding

class RGATEncoder(torch.nn.Module):
    def __init__(self, num_edge_types, in_channels, out_channels, edge_dim, gamma):
        super(RGATEncoder, self).__init__()
        self.gamma = gamma
        self.out_channels = out_channels
        self.conv1 = RGATConv(in_channels, out_channels, num_edge_types, heads=4, dropout=0.1,
                              edge_dim=edge_dim, gamma=self.gamma, concat=False)
        self.conv2 = RGATConv(out_channels, out_channels, num_edge_types, heads=1, concat=True,
                              dropout=0.1, edge_dim=edge_dim, gamma=self.gamma)
        self.out_linear = torch.nn.Linear(self.out_channels, self.out_channels)

    def forward(self, x, edge_index, edge_type, edge_attr):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        x = F.elu(x)
        return self.out_linear(x)

class AttentionPredictor(nn.Module):
    def __init__(self, hidden_dim, vocab_size, dropout=0.1):
        super(AttentionPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        # Attention mechanism components
        self.W_q = nn.Linear(hidden_dim, hidden_dim) # Query (BF node)
        self.W_k = nn.Linear(hidden_dim, hidden_dim) # Key (All nodes)
        self.W_v = nn.Linear(hidden_dim, hidden_dim) # Value (All nodes)
        self.scale = math.sqrt(hidden_dim)
        
        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # Input: [bf_embed, attention_context]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, all_node_embeds, bf_node_indices, keyword_indices=None): # keyword_indices unused here
        
        bf_embeds = all_node_embeds[bf_node_indices] # Shape: [num_bf_nodes, hidden_dim]
        if bf_embeds.numel() == 0:
            return torch.empty(0, self.mlp[-1].out_features, device=all_node_embeds.device)
            
        # Prepare for attention: Q from BF nodes, K/V from all nodes
        Q = self.W_q(bf_embeds)             # Shape: [num_bf_nodes, hidden_dim]
        K = self.W_k(all_node_embeds)       # Shape: [num_all_nodes, hidden_dim]
        V = self.W_v(all_node_embeds)       # Shape: [num_all_nodes, hidden_dim]

        # Calculate scaled dot-product attention scores
        # attn_scores shape: [num_bf_nodes, num_all_nodes]
        attn_scores = torch.matmul(Q, K.transpose(0, 1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Calculate context vector using weighted sum of Values
        # context shape: [num_bf_nodes, hidden_dim]
        context = torch.matmul(attn_probs, V)

        # Concatenate BF node embedding and its specific context vector
        combined_features = torch.cat([bf_embeds, context], dim=1) # Shape: [num_bf_nodes, hidden_dim * 2]

        # Final prediction
        output_logits = self.mlp(combined_features) # Shape: [num_bf_nodes, vocab_size]
        return output_logits

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        encoder_outputs = encoder_outputs.unsqueeze(0)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        energy = energy.squeeze(0)
        attention = torch.sum(self.v * energy, dim=1)
        return F.softmax(attention, dim=0)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input)).squeeze(0)
        a = self.attention(hidden, encoder_outputs).unsqueeze(0)
        weighted = torch.bmm(a.unsqueeze(1), encoder_outputs.unsqueeze(0)).squeeze(1)
        gru_input = torch.cat((embedded, weighted), dim=1)
        output, hidden = self.gru(gru_input.unsqueeze(0), hidden.unsqueeze(0))
        prediction = self.out(output.squeeze(0))
        return prediction, hidden.squeeze(0)

class AFGeneratorModel(nn.Module):
    def __init__(self, num_edge_types, in_channels, hidden_dim, edge_dim, gamma, vocab_size):
        super(AFGeneratorModel, self).__init__()
        self.encoder = RGATEncoder(num_edge_types, in_channels, hidden_dim, edge_dim, gamma)
        self.predictor = AttentionPredictor(hidden_dim, vocab_size)
        self.decoder = Decoder(vocab_size, in_channels, hidden_dim)
        self.hidden_dim = hidden_dim
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim) 

    def forward(self, data):
        all_node_embeds = self.encoder(data.x, data.edge_index, data.edge_type, data.edge_attr)
        branch_logits = self.predictor(all_node_embeds, data.bf_node_indices)
        
        decoder_outputs = []
        
        if not hasattr(data, 'af_sequences_input') or data.af_sequences_input is None or data.af_sequences_input.numel() == 0:
            return branch_logits, decoder_outputs

        af_sequences_input = data.af_sequences_input

        for i in range(len(af_sequences_input)):
            # Handle potential batch dimension inconsistency for af_contexts
            context_bf_node_indices_list = data.af_contexts[0][i] if isinstance(data.af_contexts, list) and len(data.af_contexts) > 0 and isinstance(data.af_contexts[0], list) else data.af_contexts[i]
            if not context_bf_node_indices_list: continue

            context_bf_node_indices = torch.tensor(context_bf_node_indices_list, dtype=torch.long, device=all_node_embeds.device)
            # Ensure indices are within bounds
            if torch.max(context_bf_node_indices) >= all_node_embeds.shape[0]: continue
            context_embeds = all_node_embeds[context_bf_node_indices]

            if context_embeds.dim() == 1: branch_context_embed = context_embeds.view(1, -1)
            elif context_embeds.dim() > 2: branch_context_embed = context_embeds.mean(dim=list(range(context_embeds.dim()-1))).view(1, -1)
            else: branch_context_embed = context_embeds.mean(dim=0).view(1, -1)
            
            # Ensure start node index is valid
            if i >= len(data.af_start_nodes_for_decoder): continue
            start_node_index = data.af_start_nodes_for_decoder[i].item()
            if start_node_index >= all_node_embeds.shape[0]: continue
            af_start_node_embed = all_node_embeds[start_node_index].view(1, -1)
            
            # Ensure dimensions match before concatenation (robustness)
            if branch_context_embed.shape[1] != self.hidden_dim or af_start_node_embed.shape[1] != self.hidden_dim:
                 branch_context_embed = branch_context_embed[:, :self.hidden_dim] if branch_context_embed.shape[1] > self.hidden_dim else torch.cat([branch_context_embed, torch.zeros(1, self.hidden_dim - branch_context_embed.shape[1], device=branch_context_embed.device)], dim=1)
                 af_start_node_embed = af_start_node_embed[:, :self.hidden_dim] if af_start_node_embed.shape[1] > self.hidden_dim else torch.cat([af_start_node_embed, torch.zeros(1, self.hidden_dim - af_start_node_embed.shape[1], device=af_start_node_embed.device)], dim=1)

            initial_hidden_input = torch.cat([branch_context_embed, af_start_node_embed], dim=1)
            hidden = torch.relu(self.fc_hidden(initial_hidden_input))

            if not hasattr(data, 'all_bf_node_indices') or data.all_bf_node_indices.numel() == 0: continue
            encoder_outputs = all_node_embeds[data.all_bf_node_indices]
            
            af_input_seq = af_sequences_input[i]
            
            seq_decoder_outputs = []
            for t in range(len(af_input_seq)):
                input_token = af_input_seq[t].unsqueeze(0)
                output, hidden = self.decoder(input_token, hidden, encoder_outputs)
                seq_decoder_outputs.append(output)

            decoder_outputs.append(torch.cat(seq_decoder_outputs, dim=0))

        return branch_logits, decoder_outputs