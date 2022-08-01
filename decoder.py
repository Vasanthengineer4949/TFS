import torch
import torch.nn as nn
from encoder import EncoderBlock
from attention import SelfAttention

class DecoderBlock(nn.Module):

    def __init__(self, embed_dim, heads, forward_exp, dropout, device):

        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(heads, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.enc_block = EncoderBlock(heads, embed_dim, dropout, forward_exp)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hid_st, key, value, src_mask, trg_mask):

        attention = self.attention(hid_st, hid_st, hid_st, trg_mask)
        norm_atten = self.norm(attention + hid_st)
        query = self.dropout(norm_atten)
        dec_out = self.enc_block(query, key, value, src_mask)

        return dec_out
    
class Decoder(nn.Module):

    def __init__(self,
                tgt_vocab_size,
                embed_dim,
                num_layers,
                device,
                heads,
                forward_exp,
                dropout,
                max_length):

        super(Decoder, self).__init__()

        self.device = device
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, heads, forward_exp, dropout, device)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, trg_token, enc_hid, src_mask, trg_mask):

        N, seq_len = trg_token.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        token_embedding = self.token_embedding(trg_token)
        position_embedding = self.position_embedding(positions)
        embedding = token_embedding + position_embedding
        embedding = self.dropout(embedding)

        for layer in self.layers:
            embedding = layer(embedding, enc_hid, enc_hid, src_mask, trg_mask)

        out = self.fc_out(embedding)

        return out
