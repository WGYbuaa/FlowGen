import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class NonGraphTransformerEncoder(nn.Module):
    """Use Transformer encoding on the ordered BF node sequence only."""

    def __init__(self, input_dim, hidden_dim, num_heads=8, num_layers=2, ffn_dim=None, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or hidden_dim * 4

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, all_bf_node_indices=None):
        all_node_embeds = self.input_proj(x)
        if all_bf_node_indices is None or all_bf_node_indices.numel() == 0:
            return all_node_embeds

        bf_embeds = all_node_embeds[all_bf_node_indices].unsqueeze(0)
        bf_embeds = self.positional_encoding(bf_embeds)
        bf_contextualized = self.encoder(bf_embeds).squeeze(0)

        fused_bf = self.fuse(torch.cat([all_node_embeds[all_bf_node_indices], bf_contextualized], dim=-1))
        contextualized_nodes = all_node_embeds.clone()
        contextualized_nodes[all_bf_node_indices] = fused_bf
        return contextualized_nodes


class TransformerBPPPredictor(nn.Module):
    """Keep BPP output shape aligned with the original implementation."""

    def __init__(self, hidden_dim, vocab_size, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, all_node_embeds, bf_node_indices, keyword_indices=None):
        bf_embeds = all_node_embeds[bf_node_indices]
        if bf_embeds.numel() == 0:
            return torch.empty(0, self.classifier[-1].out_features, device=all_node_embeds.device)

        if keyword_indices is not None and keyword_indices.numel() > 0:
            keyword_context = all_node_embeds[keyword_indices].mean(dim=0, keepdim=True)
        else:
            keyword_context = bf_embeds.mean(dim=0, keepdim=True)

        keyword_context = keyword_context.expand(bf_embeds.size(0), -1)
        fused_features = torch.cat([bf_embeds, keyword_context], dim=-1)
        return self.classifier(fused_features)


class TransformerAFDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads=8, num_layers=2, ffn_dim=None, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or hidden_dim * 4
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_tokens, memory):
        tgt = self.token_embedding(input_tokens) * math.sqrt(self.hidden_dim)
        tgt = self.positional_encoding(tgt)
        seq_len = input_tokens.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_tokens.device, dtype=torch.bool), diagonal=1
        )
        decoded = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal_mask)
        return self.output_layer(decoded)


class NonGraphTransformerModel(nn.Module):
    """Transformer baseline aligned with the original BPP and AFGen interfaces."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        vocab_size,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        ffn_dim=None,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = NonGraphTransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.predictor = TransformerBPPPredictor(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            dropout=dropout,
        )
        self.decoder = TransformerAFDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        self.condition_from_context = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.log_var_branch = nn.Parameter(torch.tensor(0.0, dtype=torch.float))
        self.log_var_decoder = nn.Parameter(torch.tensor(0.0, dtype=torch.float))

    def encode_subgraph(self, data):
        all_bf_node_indices = getattr(data, "all_bf_node_indices", None)
        return self.encoder(data.x, all_bf_node_indices)

    def build_condition_token(self, all_node_embeds, start_node_idx, context_indices):
        start_node_embed = all_node_embeds[start_node_idx].view(1, -1)
        if context_indices is None or context_indices.numel() == 0:
            return None
        context_embed = all_node_embeds[context_indices].mean(dim=0, keepdim=True)
        return self.condition_from_context(torch.cat([context_embed, start_node_embed], dim=-1))

    def build_decoder_memory(self, all_node_embeds, condition_token):
        return torch.cat([condition_token, all_node_embeds], dim=0).unsqueeze(0)
