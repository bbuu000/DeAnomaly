from layers.Myformer_EncDec import EncoderLayer, Encoder, AttentionLayer1, AttentionLayer2
import torch.nn as nn
import torch
from layers.Attention import FourierAttention, TemporalAttention




class TS_Model(nn.Module):
    """
    normal pattern learning
    """
    def __init__(self, seq_len, num_nodes, d_model):
        super(TS_Model, self).__init__()
        # Attention
        frequencyAttention = FourierAttention(guidance_num=25)

        temporalAttention = TemporalAttention(False, attention_dropout=0.1, guidance_num=25)


        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer1(temporalAttention, d_model=d_model // 2, guidance_num=25),
                    AttentionLayer2(frequencyAttention, d_model=d_model // 2, guidance_num=25),
                    d_model=d_model,
                    dropout=0.1,
                    seq_len=seq_len,
                    num_nodes=num_nodes
                )
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            d_model=d_model,
            out_dim=num_nodes
        )



    def forward(self, x):
        output = self.encoder(x)
        return output

