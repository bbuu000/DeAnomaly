import torch
import torch.nn as nn
import torch.nn.functional as F
from .NMT import NMA, AttentionLayer
from .embed import DataEmbedding
from utils.utils import calc_diffusion_step_embedding



def swish(x):
    return x * torch.sigmoid(x)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, attn_mask=None):
        csp_x = x.clone()
        new_x = self.attention(
            csp_x, csp_x, csp_x,
            attn_mask=attn_mask
        )
        csp_x = csp_x + self.dropout(new_x)
        csp_x = self.norm1(csp_x)

        return csp_x


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for attn_layer in self.attn_layers:
            x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class AD_Model(nn.Module):
    def __init__(self, enc_in, c_out, d_model=128, n_heads=4, d_ff=128,
                 diffusion_step_embed_dim_in=128, diffusion_step_embed_dim_mid=128,
                 diffusion_step_embed_dim_out=128, dropout=0.1, output_attention=True):
        super(AD_Model, self).__init__()
        self.output_attention = output_attention
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in
        self.d_ff = d_ff
        self.embedding = DataEmbedding(enc_in * 3, d_model, dropout)
        self.kernel_size = 3
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(NMA(attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    dropout=dropout
                )
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)
        self.fc_t3 = nn.Linear(diffusion_step_embed_dim_out, d_ff)
        self.projection = nn.Conv1d(d_model, c_out, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2)


    def forward(self, x):
        noise, mask, diffusion_steps = x
        mask = mask.permute(0, 2, 1)
        x = noise.permute(0, 2, 1)

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))
        part_t = self.fc_t3(diffusion_step_embed)
        B, C = part_t.shape
        part_t = part_t.view([B, 1, C])

        c = torch.zeros_like(x)
        enc_out = torch.cat((x, mask, c), dim=2)

        enc_out = self.embedding(enc_out) + part_t
        enc_out = self.encoder(enc_out)
        enc_out = enc_out.permute(0, 2, 1)
        enc_out = self.projection(enc_out)

        c = (enc_out.permute(0, 2, 1) - x) ** 2
        enc_out2 = torch.cat((x, mask, c), dim=2)

        enc_out2 = self.embedding(enc_out2) + part_t
        enc_out2 = self.encoder(enc_out2)
        enc_out2 = enc_out2.permute(0, 2, 1)
        enc_out2 = self.projection(enc_out2)

        return enc_out, enc_out2
