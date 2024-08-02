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
        self.conv = nn.Conv1d(in_channels=d_model // 2, out_channels=d_model // 2, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.norm1 = nn.LayerNorm(d_model // 2)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SELU()

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model * 2, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=d_model * 2, out_channels=d_model, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.activation2 = F.gelu


    def forward(self, x, attn_mask=None):
        split_x = torch.split(x, x.shape[2] // 2, dim=2)
        csp_x = split_x[1].clone()
        norm_x = split_x[0].clone()
        norm_x = self.conv(norm_x.permute(0, 2, 1))
        norm_x = self.activation(norm_x)
        norm_x = norm_x.transpose(1, 2)
        new_x = self.attention(  # >> [batch_size, window_size, 512]
            csp_x, csp_x, csp_x,
            attn_mask=attn_mask
        )
        csp_x = csp_x + self.dropout(new_x)  # 残差连接
        csp_x = self.norm1(csp_x)  # >> [batch_size, window_size, 512]
        x_cat = torch.cat((csp_x, norm_x), 2)
        residual = x.clone()
        y = self.activation2(self.conv1(x_cat.permute(0, 2, 1)))
        y = self.dropout(self.conv2(y).permute(0, 2, 1))
        y = self.norm2(y + residual)

        return y


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [batch_size, window_size, 512]
        for attn_layer in self.attn_layers:
            x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x  # x [batch_size, window_size, 512]


class AD_Model(nn.Module):
    def __init__(self, enc_in, c_out, d_model=128, n_heads=4, d_ff=128,
                 diffusion_step_embed_dim_in=128, diffusion_step_embed_dim_mid=128,
                 diffusion_step_embed_dim_out=128, dropout=0.1, output_attention=True):
        super(AD_Model, self).__init__()
        self.output_attention = output_attention
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in
        self.d_ff = d_ff
        # Encoding
        self.embedding = DataEmbedding(enc_in * 3, d_model, dropout)
        self.kernel_size = 3
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(NMA(attention_dropout=dropout, output_attention=output_attention),
                        d_model//2, n_heads),
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
        # noise: [batch_size, dimensions, window_size]  加噪音之后的时间序列
        # diffusion_steps: [batch_size, 1]  batch_size中的每一个多变量时间序列扩散的步长
        noise, mask, diffusion_steps = x
        mask = mask.permute(0, 2, 1)
        x = noise.permute(0, 2, 1)  # [batch_size, window_size, dimensions]

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))   # >>  [batch_size, 512]
        part_t = self.fc_t3(diffusion_step_embed)  # >>  [batch_size, 512]  Embed a diffusion step $t$ into a higher dimensional space
        B, C = part_t.shape
        part_t = part_t.view([B, 1, C])   # >>  [batch_size, 1, 512]

        c = torch.zeros_like(x)
        enc_out = torch.cat((x, mask, c), dim=2)

        enc_out = self.embedding(enc_out) + part_t  # >>  [batch_size, window_size, 512]
        enc_out = self.encoder(enc_out)  # >>  [batch_size, window_size, 512]
        enc_out = enc_out.permute(0, 2, 1)  # >>  [batch_size, 512, window_size]
        enc_out = self.projection(enc_out)  # >>  [batch_size, dimensions, window_size]

        c = (enc_out.permute(0, 2, 1) - x) ** 2
        enc_out2 = torch.cat((x, mask, c), dim=2)

        enc_out2 = self.embedding(enc_out2) + part_t  # >>  [batch_size, window_size, 512]
        enc_out2 = self.encoder(enc_out2)  # >>  [batch_size, window_size, 512]
        enc_out2 = enc_out2.permute(0, 2, 1)  # >>  [batch_size, 512, window_size]
        enc_out2 = self.projection(enc_out2)  # >>  [batch_size, dimensions, window_size]

        return enc_out, enc_out2
