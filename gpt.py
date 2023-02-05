"""Reimplementation of Andrej Karpathy's nanogpt.

Some constants:
B: Batch dim, T - time dimension (here max of block size), C - channel or embed dim
"""
import torch
import torch._dynamo.config
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from dataclasses import dataclass, field, is_dataclass


@dataclass(eq=False)
class AttentionHead(nn.Module):

    """A single attention head implementation.

    This takes in the batch x block_sz of word embeddings or output of the previous
    decoder layer (B x T x C) and then produces (B x T x embed_dim) output. Where
    each (b, t) outputs a weighted value vector. The (b, t) example looks at the
    previous timesteps (b, 0:t-1) computes attention scores ie.,
    softmax(query[t]@key[1:t-1]) and computes a weighted sum of the value vectors from
    1:t-1.
    """

    input_dim: int
    output_dim: int
    block_size: int
    dropout_frac: float

    def __post_init__(self):
        super().__init__()
        lin_kwargs = dict(
            in_features=self.input_dim, out_features=self.output_dim, bias=False
        )
        self.key = nn.Linear(**lin_kwargs)
        self.query = nn.Linear(**lin_kwargs)
        self.value = nn.Linear(**lin_kwargs)
        self.register_buffer(
            "mask", torch.tril(torch.ones(self.block_size, self.block_size))
        )
        # Keep the initial softmax fairly diffused so that we look at all the past
        for nm, param in self.key.named_parameters():
            if nm != "weight":
                continue
            param.data *= 0.01
        self.dropout = nn.Dropout(self.dropout_frac)

    def forward(self, X):
        # X : B x T x C, outputs B x T x E (embed_dim)
        B, T, C = X.shape
        keys, queries, values = self.key(X), self.query(X), self.value(X)  # B,T,E
        attn_logit = (
            torch.einsum("ijk,ilk->ijl", queries, keys) / self.output_dim**0.5
        )  # B x T x T
        attn_logit[:, self.mask == 0] = -torch.inf
        affinities = self.dropout(F.softmax(attn_logit, dim=2))
        return affinities @ values  # B,T,T @ B,T,E = B,T,E


@dataclass(eq=False)
class MultiHeadedAttention(nn.Module):

    """This takes in the batch x block_sz of word embeddings or output of the previous
    decoder layer (B x T x C) and then produces (B x T x (nhead*(out_dim/nhead)))
    output.

    We create nhead attention heads each mapping C -> (out_dim/nhead) space. The
    """

    nhead: int
    embed_dim: int
    block_size: int
    dropout_frac: float

    def __post_init__(self):
        super().__init__()
        assert (
            self.embed_dim % self.nhead == 0
        ), "Embedding dim not divisible num attention heads"
        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    input_dim=self.embed_dim,
                    output_dim=self.embed_dim // self.nhead,
                    block_size=self.block_size,
                    dropout_frac=self.dropout_frac,
                )
                for _ in range(self.nhead)
            ],
        )
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout_lyr = nn.Dropout(self.dropout_frac)

    def forward(self, X):
        # concat nhead of dims B x T x C/nhead
        attn = torch.cat([h(X) for h in self.heads], dim=-1)
        return self.dropout_lyr(self.projection(attn))


@dataclass(eq=False)
class MLP(nn.Module):

    embed_dim: int
    hidden_dim: int
    nlayers: int
    dropout_frac: float

    def __post_init__(self):
        super().__init__()
        embed_dim, hidden_dim = self.embed_dim, self.hidden_dim
        first_layer = [
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
        ]
        hidden_layers = []
        for _ in range(self.nlayers - 1):
            hidden_layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        last_layer = [nn.Linear(hidden_dim, embed_dim), nn.Dropout(self.dropout_frac)]
        layers = first_layer + hidden_layers + last_layer
        self.sequential = nn.Sequential(*layers)

    def forward(self, X):
        return self.sequential(X)


@dataclass(eq=False)
class Block(nn.Module):

    embed_dim: int
    block_size: int
    num_attn_head: int
    mlp_hidden_layers: int
    mlp_hidden_dim: int
    dropout_frac: float

    def __post_init__(self):
        super().__init__()
        self.sa = MultiHeadedAttention(
            nhead=self.num_attn_head,
            embed_dim=self.embed_dim,
            block_size=self.block_size,
            dropout_frac=self.dropout_frac,
        )
        self.ln1 = nn.LayerNorm(normalized_shape=self.embed_dim)
        self.mlp = MLP(
            embed_dim=self.embed_dim,
            hidden_dim=self.mlp_hidden_dim,
            nlayers=self.mlp_hidden_dim,
            dropout_frac=self.dropout_frac,
        )
        self.ln2 = nn.LayerNorm(normalized_shape=self.embed_dim)

    def forward(self, X):
        attn = X + self.sa(self.ln1(X))
        out = attn + self.mlp(self.ln2(attn))
        return out


@dataclass(eq=False)
class Transformer(nn.Module):

    vocab_sz: int
    embed_dim: int
    block_size: int
    num_attn_head: int
    mlp_hidden_layers: int
    mlp_hidden_dim: int
    num_blocks: int
    dropout_frac: float
    device: str

    def __post_init__(self):
        super().__init__()
        vocab_sz, embed_dim = self.vocab_sz, self.embed_dim
        block_size = self.block_size
        self.tkn_embed_tbl = nn.Embedding(
            num_embeddings=vocab_sz, embedding_dim=embed_dim
        )
        self.pos_embed_tbl = nn.Embedding(
            num_embeddings=block_size, embedding_dim=embed_dim
        )
        block_kwargs = dict(
            embed_dim=embed_dim,
            block_size=block_size,
            num_attn_head=self.num_attn_head,
            mlp_hidden_dim=self.mlp_hidden_dim,
            mlp_hidden_layers=self.mlp_hidden_layers,
            dropout_frac=self.dropout_frac,
        )
        self.blocks = nn.Sequential(
            *[Block(**block_kwargs) for _ in range(self.num_blocks)],  # type: ignore
        )
        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.logit_layer = nn.Linear(in_features=embed_dim, out_features=vocab_sz)
        # make predictions less confident at the beginning so that all
        # classes about equal probability.
        self.logit_layer.weight.data = self.logit_layer.weight * 0.01

    def forward(self, X):
        """Returns the forward pass and NLL loss"""
        B, T = X.shape
        tkn_embed = self.tkn_embed_tbl(X)  # B x T x C
        pos_embed = self.pos_embed_tbl(torch.arange(T, device=self.device))  # T x C
        wv = tkn_embed + pos_embed  # B x T x C
        dec_out = self.blocks(wv)
        normd_dec_out = self.ln_f(dec_out)
        out = self.logit_layer(normd_dec_out)  # B x T x V
        return out


#######################################################################################
# Generate examples from models
#######################################################################################
def gen_examples(
    model: nn.Module,
    max_len: int,
    block_size: int,
    num_examples: int,
    device: str,
) -> torch.Tensor:
    """Generates examples from the model. We always begin with a newline character.
    Logic: We always take the prediction of the last time dim, in the block. We add
    the
    """
    # B x T: zero corresponds to the newline character
    # This always contains the current block size of context
    cur_block = torch.zeros(num_examples, block_size, dtype=torch.int, device=device)
    # This will be appended with the next prediction continuously
    outputs = torch.zeros(num_examples, 1, dtype=torch.int, device=device)
    for _ in range(max_len):
        # B x C
        logits = model(cur_block)
        # B x C: we only look at the last time step to sample the next token
        probas = F.softmax(logits[:, -1, :], dim=1)
        # B x 1
        next_ix = torch.multinomial(probas.squeeze(), num_samples=1)
        outputs = torch.concat((outputs, next_ix), axis=1)  # type: ignore
        cur_block = torch.concat((cur_block[:, 1:], next_ix), axis=1)  # type: ignore
    return outputs


def print_examples(
    model: nn.Module,
    max_len: int,
    ixtoc: dict,
    block_size: int,
    num_examples: int,
    device: str,
    to_replace_nl=False,
):
    examples = gen_examples(
        model=model,
        max_len=max_len,
        block_size=block_size,
        num_examples=num_examples,
        device=device,
    )
    for ex in examples:
        print("".join(decode(ex, ixtoc=ixtoc, to_replace_nl=to_replace_nl)))
        print("-----------------------------------------------------------")


def decode(x, ixtoc, to_replace_nl=False):
    txt = "".join(ixtoc[xi.item()] for xi in x)
    if to_replace_nl:
        txt = txt.replace("\n", "<N>")
    return txt
