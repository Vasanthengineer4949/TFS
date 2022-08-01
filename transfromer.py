import torch
import torch.nn as nn
from attention import SelfAttention
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):

    def __init__(self,
                src_vocab_size,
                tgt_vocab_size,
                src_pad_idx,
                tgt_pad_idx,
                embed_dim = 256,
                num_layers = 6,
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                heads = 8,
                forward_exp = 4,
                dropout = 0.2,
                max_length = 100
        ):
        
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, 
                            embed_dim, 
                            num_layers, 
                            device, 
                            heads, 
                            forward_exp, 
                            dropout,
                            max_length)

        self.decoder = Decoder(tgt_vocab_size,
                            embed_dim,
                            num_layers,
                            device,
                            heads,
                            forward_exp,
                            dropout,
                            max_length)

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def create_source_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        src_mask = src_mask.to(self.device)
        return src_mask

    def create_target_mask(self, tgt):
        N, tgt_len = tgt.shape
        tgt_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), dtype=torch.uint8)).expand(
                N, 1, tgt_len, tgt_len
                )
        tgt_mask = tgt_mask.to(self.device)
        return tgt_mask
        
    def forward(self, src, tgt):

        src_mask = self.create_source_mask(src)
        tgt_mask = self.create_target_mask(tgt)

        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
            
        return dec_out
        



