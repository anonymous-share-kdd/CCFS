import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from hyper_connections import HyperConnections



class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        num_residual_streams = 4
    ):
        super().__init__()

        init_hyper_conn, self.expand_streams, self.reduce_streams = HyperConnections.get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                init_hyper_conn(dim = dim, branch = Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                init_hyper_conn(dim = dim, branch = FeedForward(dim, dropout = ff_dropout)),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        x = self.expand_streams(x)

        for attn, ff in self.layers:
            x, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = ff(x)

        x = self.reduce_streams(x)

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)


class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

class CategoricalTokenizer(nn.Module):
    def __init__(self, categories, dim: int, num_special_tokens: int = 1):
        super().__init__()
        assert len(categories) > 0 and all(c > 0 for c in categories)
        self.num_categories = len(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = sum(categories) + num_special_tokens
        offsets = torch.tensor([0] + list(categories), dtype=torch.long)
        offsets = F.pad(offsets[1:], (1, 0), value=num_special_tokens).cumsum(dim=0)[:-1]
        self.register_buffer("offsets", offsets)
        self.emb = nn.Embedding(total_tokens, dim)

    def forward(self, x_cat: torch.Tensor):
        x = x_cat + self.offsets
        return self.emb(x)

class SimpleMLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            if i > 0:
                layers.append(nn.LayerNorm(dims[i]))
                layers.append(act if act else nn.ReLU())
                layers.append(nn.Dropout(0.1))
            layers.append(nn.Linear(dims[i], dims[i+1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class FeatureTokenizer(nn.Module):
    def __init__(self, num_columns, hidden_size):
        super().__init__()
        self.num_columns = num_columns


        self.weight = nn.Parameter(torch.randn(num_columns, hidden_size))
        self.bias = nn.Parameter(torch.randn(num_columns, hidden_size))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):

        x_expanded = x.unsqueeze(-1)

        x_emb = x_expanded * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        return x_emb

class TabularEncoder(nn.Module):
    def __init__(self, num_columns: int, args):
        super(TabularEncoder, self).__init__()

        self.hidden_size = getattr(args, 'hidden_size', 768)
        self.num_heads = getattr(args, 'num_heads', 16)
        self.num_layers = getattr(args, 'num_layers', 5)
        self.dropout_prob = getattr(args, 'dropout', 0.1)
        self.activation = getattr(args, 'activation', 'gelu')
        self.ffn_dim = args.ffn_dim
        self.feature_tokenizer = FeatureTokenizer(num_columns, self.hidden_size)
        self.embedding_activation = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU() if self.activation == 'gelu' else nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        nn.init.normal_(self.cls_token, std=0.02)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            activation='relu',
            batch_first=True,
            norm_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)


    def forward(self, x_numerical: torch.Tensor):
        batch_size = x_numerical.shape[0]

        feat_emb = self.feature_tokenizer(x_numerical)
        feat_emb = self.embedding_activation(feat_emb)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        combined_emb = torch.cat((cls_tokens, feat_emb), dim=1)

        # combined_emb = combined_emb + self.col_identity_embedding

        out = self.transformer_encoder(combined_emb)

        return out


class CrossAttentionSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim=768):
        super(CrossAttentionSelector, self).__init__()
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, h_text, h_tab):

        Q = self.W_q(h_text)
        K = self.W_k(h_tab)

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.squeeze(1)
        gate = torch.sigmoid(scores * self.temperature)
        return gate

class CCFS(nn.Module):
    def __init__(self,
                 num_columns: int,
                 task_type: str,
                 args,
                 output_dim: int = 1):
        super(CCFS, self).__init__()
        self.task_type = task_type

        self.hidden_size = getattr(args, 'hidden_size', 768)
        self.dropout_prob = getattr(args, 'dropout', 0.1)

        self.tab_encoder = TabularEncoder(num_columns, args)

        self.text_projector = nn.Sequential(
            nn.Linear(768, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )

        self.selector = CrossAttentionSelector(input_dim=self.hidden_size, hidden_dim=self.hidden_size)

        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_size // 2, output_dim)
        )

    def forward(self, x_tab, x_text, return_gate=False):

        transformer_out = self.tab_encoder(x_tab)

        cls_out = transformer_out[:, 0, :]
        feat_out = transformer_out[:, 1:, :]

        h_text = self.text_projector(x_text).unsqueeze(1)

        gate = self.selector(h_text, feat_out)

        feat_selected = feat_out * gate.unsqueeze(-1)


        eps = 1e-9
        feat_pooled = feat_selected.sum(dim=1) / (gate.sum(dim=1, keepdim=True) + eps)
        h_final = cls_out + feat_pooled

        logits = self.head(h_final)

        if return_gate:
            return logits, gate
        return logits

