# 文件: src/model.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        x = torch.matmul(attn_weights, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.out(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src = self.norm1(src)
        src = src + self.dropout(self.self_attn(_src, _src, _src, src_mask))
        _src = self.norm2(src)
        src = src + self.dropout(self.feed_forward(_src))
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg = self.norm1(trg)
        trg = trg + self.dropout(self.self_attn(_trg, _trg, _trg, trg_mask))
        _trg = self.norm2(trg)
        trg = trg + self.dropout(self.cross_attn(_trg, enc_src, enc_src, src_mask))
        _trg = self.norm3(trg)
        trg = trg + self.dropout(self.feed_forward(_trg))
        return trg

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, use_pos_encoding):
        super().__init__()
        self.use_pos_encoding = use_pos_encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        if self.use_pos_encoding:
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        if self.use_pos_encoding:
            src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, use_pos_encoding):
        super().__init__()
        self.use_pos_encoding = use_pos_encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        if self.use_pos_encoding:
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.embedding(trg) * math.sqrt(self.d_model)
        if self.use_pos_encoding:
            trg = self.pos_encoder(trg)
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        return self.fc_out(trg)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, use_pos_encoding=True):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, use_pos_encoding)
        self.decoder = Decoder(trg_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, use_pos_encoding)

    def make_src_mask(self, src, pad_idx):
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg, pad_idx):
        trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
        return trg_pad_mask & trg_sub_mask

    def forward(self, src, trg, src_pad_idx, trg_pad_idx):
        src_mask = self.make_src_mask(src, src_pad_idx)
        trg_mask = self.make_trg_mask(trg, trg_pad_idx)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output