import torch
import torch.nn as nn
from zmq import device

from transfromer import Transformer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src = torch.randint(0, 10, (1, 10)).to(device)
    tgt = torch.randint(0, 10, (1, 10)).to(device)

    src_pad_idx = 0
    tgt_pad_idx = 0

    src_vocab_size = 10
    tgt_vocab_size = 10

    model =  Transformer(src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx).to(device)

    output = model(src, tgt[:, :-1])
    print(output.shape)
    print(output)

