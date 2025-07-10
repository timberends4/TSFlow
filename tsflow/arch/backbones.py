import math

import torch
from einops import rearrange, repeat
from linear_attention_transformer import LinearAttentionTransformer
from torch import nn
from torchtyping import TensorType
from typeguard import typechecked
import torch.nn.functional as F

from tsflow.arch.s4 import S4



def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

    return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=1):
        super().__init__()
        self.channels = config["channels"]

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, t_emb):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = t_emb

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)


    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        if diffusion_emb.ndim == 3:
            diffusion_emb = diffusion_emb.squeeze(1 if diffusion_emb.shape[1] == 1 else 2)
        # Now we must have exactly [B,C]
        assert diffusion_emb.ndim == 2 and diffusion_emb.shape == (B, channel), \
            f"diffusion_emb must be [B,C], got {tuple(diffusion_emb.shape)}"

        # Unsqueeze on the *last* axis → [B,C,1]
        diffusion_emb = diffusion_emb.unsqueeze(-1)
        assert diffusion_emb.shape == (B, channel, 1), \
            f"after unsqueeze, diffusion_emb must be [B,C,1], got {tuple(diffusion_emb.shape)}"
        
        y = x + diffusion_emb

        assert y.shape == (B, channel, K * L), \
            f"y must be [B,C,K*L] before forward_time, got {tuple(y.shape)}"
        
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
    

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class S4Layer(nn.Module):
    def __init__(
        self,
        d_model,
        dropout=0.0,
        bidirectional=True,
    ):
        super().__init__()
        self.layer = S4(
            d_model=d_model,
            d_state=128,
            bidirectional=bidirectional,
            dropout=dropout,
            transposed=True,
            postact=None,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        """
        Input x is shape (B, d_input, L)
        """
        z = x
        # Prenorm
        z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)
        # Apply layer: we ignore the state input and output for training
        z, _ = self.layer(z)
        # Dropout on the output of the layer
        z = self.dropout(z)
        # Residual connection
        x = z + x
        return x, None

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def step(self, x, state, **kwargs):
        z = x
        # Prenorm
        z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)
        # Apply layer
        z, state = self.layer.step(z, state, **kwargs)
        # Residual connection
        x = z + x
        return x, state


class S4Block(nn.Module):
    def __init__(
        self,
        d_model,
        dropout=0.0,
        expand=2,
        num_features=0,
        setting="univariate",
        bidirectional=True,
        layer=0,
        target_dim=1,
    ):
        super().__init__()
        self.s4block = S4Layer(d_model, bidirectional=bidirectional)
        if target_dim > 1:
            self.transformer_layer = LinearAttentionTransformer(
                dim=d_model,
                depth=1,
                heads=8,
                max_seq_len=256,
                n_local_attn_heads=0,
                local_attn_window_size=0,
            )
        self.time_linear = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.out_linear1 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.out_linear2 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.feature_encoder = nn.Conv2d(num_features, d_model, kernel_size=1)
        self.setting = setting
        self.target_dim = target_dim

    def forward(self, x, t, features=None, feat_embeddings=None):
        b, c, k, l = x.shape
        t = self.time_linear(t)
        t = rearrange(t, "b 1 c -> b c 1 1")
        s4_in = x + t
        out, _ = self.s4block(rearrange(s4_in, "b c k l -> (b k) c l"))
        if self.target_dim > 1:
            feature_in = rearrange(out, "(b k) c l -> (b l) k c", k=k, l=l)
            out = self.transformer_layer(feature_in)
            out = rearrange(out, "(b l) k c -> b c k l", l=l)
        else:
            out = rearrange(out, "(b k) c l -> b c k l", k=k)

        if features is not None:  # B C K L
            feature_out = self.feature_encoder(features)
            out = out + feature_out

        out = self.tanh(out) * self.sigm(out)
        out1 = self.out_linear1(out)
        out2 = self.out_linear2(out)
        return out1 + x, out2


class BackboneModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        step_emb: int,
        num_residual_blocks: int,
        num_features: int,
        target_dim: int = 1,
        residual_block: str = "s4",
        dropout: float = 0.0,
        bidirectional: bool = True,
        init_skip: bool = True,
        feature_skip: bool = True,
    ):
        super().__init__()
        if residual_block == "s4":
            residual_block = S4Block
        else:
            raise ValueError(f"Unknown residual block {residual_block}")
        self.input_init = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        self.time_init = nn.Sequential(
            nn.Linear(step_emb, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.out_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        def get_residual_blocks(hidden_dim, num_residual_blocks):
            residual_blocks = []
            for i in range(num_residual_blocks):
                residual_blocks.append(
                    residual_block(
                        hidden_dim,
                        num_features=num_features + feature_skip + (16 if target_dim > 1 else 0),
                        bidirectional=bidirectional,
                        dropout=dropout,
                        layer=i,
                        target_dim=target_dim,
                    )
                )
            return nn.ModuleList(residual_blocks)

        self.residual_blocks = get_residual_blocks(hidden_dim, num_residual_blocks)
        self.step_embedding = SinusoidalPositionEmbeddings(step_emb)
        self.init_skip = init_skip
        self.feature_skip = feature_skip
        self.feature_embedd = nn.Embedding(target_dim, 16)

    @typechecked
    def forward(
        self,
        t: TensorType[float],
        x_in: TensorType[float, "batch", "length", "num_series"],
        features: TensorType[float, "batch", "length", "num_series", "num_features"] | None = None,
        args: dict | None = None,
    ) -> TensorType[float, "batch", "length", "num_series"]:
        # if not self.training:
        B, L, K = x_in.shape
        x = x_in.unsqueeze(-1)
        if self.feature_skip and features is not None:
            features = torch.cat([features, x_in.unsqueeze(-1)], dim=-1)
        if len(t.shape) == 0:
            t = repeat(t, " -> b 1", b=x.shape[0])
        else:
            t = t[..., 0]
        t = self.step_embedding(t * 10000)

        feat_embeddings = None
        if features is not None:
            features = features.transpose(-1, 1)  # B, C, K, L

            if K > 1:
                feat_embeddings = repeat(
                    self.feature_embedd(torch.arange(K, device=features.device)),
                    "K C -> B C K L",
                    B=B,
                    L=L,
                )
                features = torch.cat([features, feat_embeddings], dim=1)

        t = self.time_init(t)
        x = self.input_init((x_in).unsqueeze(-1))  # B, L, H, 1   # B, L ,K, H
        x = x.transpose(-1, 1)  # B, H, 1, L  # B, H, K, L
        skips = []
        # univariate
        for i, layer in enumerate(self.residual_blocks):
            x, skip = layer(x, t, features)
            skips.append(skip)
        skip = rearrange(torch.stack(skips).sum(0), "b c k l -> b l k c")
        out = self.out_linear(skip)[..., 0]

        if self.init_skip:
            out = out - x_in
        return out


import torch
from torch import nn, Tensor
from einops import rearrange, repeat
from typeguard import typechecked


class BackboneModelMultivariate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        step_emb: int,
        num_residual_blocks: int,
        num_features: int,
        target_dim: int = 1,
        residual_block: str = "s4",
        dropout: float = 0.0,
        bidirectional: bool = True,
        init_skip: bool = True,
        feature_skip: bool = True,
    ):
        super().__init__()
        if residual_block == "s4":
            residual_block = S4Block
        else:
            raise ValueError(f"Unknown residual block {residual_block}")
        self.input_init = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())

        self.time_init = nn.Sequential(
            nn.Linear(step_emb, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.out_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # compute feature channels for S4 blocks: 1 transformer + emb_dim
        
        self.emb_dim = 16
        feat_channels = 1 + self.emb_dim
        # residual S4 blocks
        self.residual_blocks = nn.ModuleList(
            [S4Block(d_model=hidden_dim,
                      num_features=feat_channels,
                      bidirectional=bidirectional,
                      dropout=dropout,
                      layer=i,
                      target_dim=target_dim)
             for i in range(num_residual_blocks)]
        )

        self.step_embedding = SinusoidalPositionEmbeddings(step_emb)
        self.init_skip = init_skip
        self.feature_skip = feature_skip

        self.feature_embedd = nn.Embedding(target_dim, self.emb_dim)

        orig_C = num_features + (1 if feature_skip else 0)
        self.cond_proj = nn.Conv2d(orig_C, hidden_dim * 2, kernel_size=1)

        self.total_emb = hidden_dim * 2
        config_trans = {"layers": 4, "channels": hidden_dim, "nheads": 8, "diffusion_embedding_dim": hidden_dim, "num_steps": 50, "is_linear": False, "side_dim":self.total_emb}

        self.transformer_model = diff_CSDI(config_trans, inputdim=1)

    @typechecked
    def forward(
        self,
        t: TensorType[float],
        x_in: TensorType[float, "batch", "length", "num_series"],
        features: TensorType[float, "batch", "length", "num_series", "num_features"] | None = None,
        args: dict | None = None,
    ) -> TensorType[float, "batch", "length", "num_series"]:

        # unpack
        B, L, K = x_in.shape
        assert x_in.ndim == 3, f"x_in must be 3-D [B,L,K], got {tuple(x_in.shape)}"

        # 1) Build time embeddings
        if t.ndim == 0:
            t_in = repeat(t, " -> b 1", b=B)
        else:
            t_in = t[..., 0]
        t_emb = self.step_embedding(t_in * 10000)  # [B, H]
        t_emb = self.time_init(t_emb)             # [B, H, 1]
        # assert t_emb.shape == (B, t_emb.shape[-1]), f"t_emb shape wrong: {tuple(t_emb.shape)}"

        # 2) Build cond_info for transformer
        cond_feats = []
        if features is not None:
            # features: [B,L,K,F] → [B,F,K,L]
            f = features.permute(0, 3, 2, 1)
            assert f.shape == (B, features.shape[3], K, L), f"permute features→{tuple(f.shape)}"
            cond_feats.append(f)

        if self.feature_skip:
            # x_in: [B,L,K] → [B,L,K,1] → permute to [B,1,K,L]
            m = x_in.unsqueeze(-1).permute(0, 3, 2, 1)
            assert m.shape == (B, 1, K, L), f"mask feats m shape wrong: {tuple(m.shape)}"
            cond_feats.append(m)

        cond_info = torch.cat(cond_feats, dim=1)      # [B, C_cond, K, L]
        assert cond_info.ndim == 4 and cond_info.shape[0] == B, f"cond_info wrong: {tuple(cond_info.shape)}"
        cond_info = self.cond_proj(cond_info)         # [B, side_dim, K, L]
        assert cond_info.ndim == 4, f"cond_proj output wrong: {tuple(cond_info.shape)}"

        # 3) Run transformer block on series dimension
        # x_for_trans: [B,1,K,L]
        x_for_trans = x_in.permute(0, 2, 1).unsqueeze(1)
        assert x_for_trans.shape == (B, 1, K, L), f"x_for_trans wrong: {tuple(x_for_trans.shape)}"
        transformed = self.transformer_model(x_for_trans, cond_info, t_emb)  # [B, K, L]
        assert transformed.shape == (B, K, L), f"transformer output wrong: {tuple(transformed.shape)}"

        # 4) Incorporate transformer output as new feature
        new_feats = transformed.unsqueeze(1)  # [B,1,K,L]
        assert new_feats.shape == (B, 1, K, L), f"new_feats wrong: {tuple(new_feats.shape)}"

        # 5) Series positional embeddings
        feat_embeddings = repeat(
            self.feature_embedd(torch.arange(K, device=x_in.device)),
            "K C -> B C K L",
            B=B, L=L,
        )
        assert feat_embeddings.shape == (B, self.feature_embedd.embedding_dim, K, L), \
            f"feat_embeddings wrong: {tuple(feat_embeddings.shape)}"

        features = torch.cat([new_feats, feat_embeddings], dim=1)  # [B, 1+emb_dim, K, L]
        assert features.ndim == 4 and features.shape[2] == K and features.shape[3] == L, \
            f"features for S4 wrong: {tuple(features.shape)}"

        # 6) Univariate S4/residual blocks on x
        x = x_in.unsqueeze(-1)          # [B, L, K, 1]
        assert x.shape == (B, L, K, 1), f"x before input_init wrong: {tuple(x.shape)}"
        x = self.input_init(x)          # [B, L, K, H]
        assert x.ndim == 4 and x.shape[3] == self.input_init[0].out_features, \
            f"x after input_init wrong: {tuple(x.shape)}"
        x = x.permute(0, 3, 2, 1)       # [B, H, K, L]
        assert x.shape == (B, x.shape[1], K, L), f"x permuted wrong: {tuple(x.shape)}"

        # 7) Residual‐S4 stack
        skips = []
        for i, layer in enumerate(self.residual_blocks):
            x, skip = layer(x, t_emb, features)
            assert x.ndim == 4 and skip.ndim == 4, \
                f"layer {i} outputs wrong dims: x={tuple(x.shape)}, skip={tuple(skip.shape)}"
            assert x.shape[2:] == (K, L) and skip.shape[2:] == (K, L), \
                f"layer {i} outputs wrong K/L: x={tuple(x.shape)}, skip={tuple(skip.shape)}"
            skips.append(skip)

        # 8) Sum & final readout
        skip_sum = torch.stack(skips).sum(0)  # [B, H, K, L]
        assert skip_sum.shape == (B, skip_sum.shape[1], K, L), f"skip_sum wrong: {tuple(skip_sum.shape)}"
        skip_sum = rearrange(skip_sum, "b c k l -> b l k c")  # [B, L, K, H]
        assert skip_sum.shape == (B, L, K, skip_sum.shape[3]), \
            f"skip_sum rearrange wrong: {tuple(skip_sum.shape)}"
        out = self.out_linear(skip_sum)[..., 0]  # [B, L, K]
        assert out.shape == (B, L, K), f"out wrong: {tuple(out.shape)}"

        # 9) Optional initial skip
        if self.init_skip:
            out = out - x_in
            assert out.shape == (B, L, K), f"out after init_skip wrong: {tuple(out.shape)}"

        return out