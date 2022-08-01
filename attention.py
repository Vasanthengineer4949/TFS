#Creating the Attention mechanism

import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def  __init__(self, heads, embed_dim):

        super(SelfAttention, self).__init__()

        self.heads = heads # Number of heads
        self.embed_dim = embed_dim # Dimension of the embeddings
        self.head_dim = embed_dim // heads # Dimension of the head's embeddings

        assert self.head_dim * heads == self.embed_dim, "Dimension mismatch" # Check if the embedding dimension is divisible by the number of heads

        # Linear layers to transform the input embeddings
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Linear layer to concatenate the head's embeddings
        self.out = nn.Linear(self.heads*self.head_dim, self.embed_dim)

    def forward(self, query, key, value, mask):

        N = query.shape[0] # Number of samples
        query_len = query.shape[1] # Length of the query embeddings
        key_len = key.shape[1] # Length of the key embeddings
        value_len = value.shape[1] # Length of the value embeddings

        # Split the query, key and value embeddings into heads
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        keys = key.reshape(N, key_len, self.heads, self.head_dim)
        values = value.reshape(N, value_len, self.heads, self.head_dim)

        querkey =  torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

        if mask is not None:
            querkey = querkey.masked_fill(mask==0, float("-1e9"))

        # Calculate the attention weights using the scaled dot product
        attention_score = torch.softmax(querkey / (self.embed_dim**(1/2)), dim=3)
        attention_values = torch.einsum('nhql,nlhd->nqhd', [attention_score, values]) # As value_len and key_len are always of same dimension we can multiply them together (v,  k -> l)
        attention_out = attention_values.reshape(N, query_len, self.heads*self.head_dim)
        out = self.out(attention_out)

        return out

