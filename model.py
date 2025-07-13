import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

import decoder_block, encoder_block, embeddings


N = 6 # number of layers
h = 8 # number of heads
d_model = 512 # embedding dimensions
d_ff = 2048 # feedforward hidden layer
context_window = 5000

class Transformer(nn.Module):
    def __init__(self, vocab_size : int, d_model : int):
        super().__init__()
        self.input_embedding = embeddings.EmbeddingLayer(vocab_size, d_model, context_window)
        self.output_embedding = embeddings.EmbeddingLayer(vocab_size, d_model, context_window)
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for _ in range(N):
            self.encoder_blocks.append(encoder_block.Encoder(d_model))
            self.decoder_blocks.append(decoder_block.Decoder(d_model))
        self.final_linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: TensorType[int], output_ids: TensorType[int]) -> TensorType[float]:
        source_embed = self.input_embedding(input_ids)
        target_embed = self.output_embedding(output_ids)
        
        encoder_output = source_embed
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output)
        
        decoder_output = target_embed
        for block in self.decoder_blocks:
            decoder_output = block(decoder_output, encoder_output)
        
        return self.final_linear(decoder_output)
