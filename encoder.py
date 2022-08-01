from turtle import forward
from attention import SelfAttention
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, heads, embed_dim, dropout, forward_exp):
        super(EncoderBlock, self).__init__()
        self.attention = SelfAttention(heads, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*forward_exp),
            nn.ReLU(),
            nn.Linear(embed_dim*forward_exp, embed_dim)
        )

    def forward(self, query, key, value, mask):
        attention_out = self.attention(query, key, value, mask)
        norm_attention_out = self.norm1(attention_out + query)
        drop_attention_out = self.dropout(norm_attention_out)
        forward = self.ff(drop_attention_out)
        out = self.norm2(drop_attention_out + forward)
        fin_attn_out = self.dropout(out)
        return fin_attn_out

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_dim,
        num_layers,
        device,
        heads,
        forward_expansion, 
        dropout, 
        max_length
    ):
        super(Encoder, self).__init__()

        self.embed_dim = embed_dim
        # Creating the embedding layer
        self.token_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.postion_embedding = nn.Embedding(max_length, embed_dim)
        # Creating the dropout layer
        self.dropout = nn.Dropout(dropout)
        # Creating the encoder layers
        self.layers = nn.ModuleList([
            EncoderBlock(heads, embed_dim, dropout, forward_expansion)
            for _ in range(num_layers)
        ])
        self.device = device

    def forward(self, src_token, mask):

        N, seq_len = src_token.shape
        # For every example in the batch it goes from  0 to seq_len
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        # Generate embeddings for the tokens and positions
        embeddings = self.token_embedding(src_token) + self.postion_embedding(positions)

        # Generate the embeddings from the encoder layers - hidden states
        for layer in self.layers:
            enc_out = layer(embeddings, embeddings, embeddings, mask) # Three are sent to the layer because the layer needs three inputs Q, K and V   

        return enc_out
