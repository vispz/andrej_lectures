"""Reimplementation of Andrej Karpathy's nanogpt.

Some constants:
B: Batch dim, T - time dimension (here max of block size), C - channel or embed dim
"""
import datetime
import functools as ft
import matplotlib.pyplot as plt
import random
import time
import torch
import torch._dynamo.config
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.tensorboard as tb

from dataclasses import dataclass, field
from tqdm import tqdm


torch.manual_seed(1338)
torch._dynamo.config.verbose = True

plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["font.size"] = 14


st_time = time.time()
#######################################################################################
# Hyperparams
#######################################################################################
INPUT_FL = "input.txt"
EVAL_FREQ = 100
EVAL_SZ = 10000
# device = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = "mps"
print(f"Running on device: {DEVICE}")


@dataclass(frozen=True)
class HyperParams:
    block_sz: int = 8
    train_frac: float = 0.9
    batch_sz: int = 32
    max_iters: int = 1_000
    learning_rate: float = 1e-3
    # if you update this also update
    embed_dim: int = 32
    dropout_frac: float = 0.0
    num_attn_heads: int = 4
    mlp_hidden_layers: int = 1
    # embed_dim * 4
    mlp_hidden_dim: int = 32 * 4
    # number of decoder transformer blocks stacked one on top of another
    num_blocks: int = 1


HYPERPARAMS = HyperParams()
dt_str = datetime.datetime.now().strftime("%Y-%m-%d#%H:%M:%S")
tb_writer = tb.SummaryWriter(
    log_dir=f"runs/mlp_mhead_attn | {dt_str} | {str(HYPERPARAMS)}"
)


#######################################################################################
# Dataset: Text to data
#######################################################################################

def txt_to_data(input_txt, ctoix, block_size):
    Xs, Ys = [], []
    input_len = len(input_txt)
    input_ixes = encode(txt=input_txt, ctoix=ctoix)
    for i in range(0, input_len - block_size, block_size):
        Xs.append(input_ixes[i : i + block_size])
        Ys.append(input_ixes[i + 1 : i + block_size + 1])
    return torch.tensor(Xs).to(DEVICE), torch.tensor(Ys).to(DEVICE)


def encode(txt, ctoix):
    return [ctoix[char] for char in txt]


def decode(x, ixtoc, to_replace_nl=False):
    txt = "".join(ixtoc[xi.item()] for xi in x)
    if to_replace_nl:
        txt = txt.replace("\n", "<N>")
    return txt


#######################################################################################
# Dataloader
#######################################################################################


def iter_dataset(Xs, Ys, batch_sz, num=None):
    num = num or torch.inf
    i = 0
    n = Xs.shape[0]
    while i < num:
        i += 1
        batch_ixes = torch.randint(n, (batch_sz,))
        yield Xs[batch_ixes], Ys[batch_ixes]


def split_data(X, Y, train_frac=0.8, to_shuffle=True):
    num_samples = X.shape[0]
    train_ix = int(num_samples * train_frac)
    ixes = list(range(num_samples))
    if to_shuffle:
        random.shuffle(ixes)
    train_ixes, val_ixes = ixes[:train_ix], ixes[train_ix:]
    print(f"Num train: {train_ix:,}\nnum validation: {len(val_ixes):,}")
    return {"train": (X[train_ixes], Y[train_ixes]), "val": (X[val_ixes], Y[val_ixes])}


#######################################################################################
# Training loop
#######################################################################################


def train_model(
    model,
    Xtrain,
    Ytrain,
    batch_sz,
    num_iters,
    optimizer,
    eval_loss_fn,
    losses: dict[str, list],  # train -> [], val -> []
    loss_calc_freq=100,
):
    for i, (Xi, Yi) in (
        pbar := tqdm(
            enumerate(
                iter_dataset(Xs=Xtrain, Ys=Ytrain, num=num_iters, batch_sz=batch_sz)
            ),
            total=num_iters,
        )
    ):
        loss = eval_model_loss(model=model, X=Xi, Ytrue=Yi)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        # Metric tracking
        if i % loss_calc_freq == 0:
            eval_losses = eval_loss_fn(model=model)
            tb_writer.add_scalars(
                main_tag="loss", tag_scalar_dict=eval_losses, global_step=i
            )
            loss_msg = ""
            for split, loss in eval_losses.items():
                losses[split].append(loss)
                loss_msg += f"{split} loss: {loss:.4f} "
            pbar.set_description(loss_msg)


@torch.no_grad()
def eval_loss(model, data_splits, eval_sz):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        X, Y = data_splits[split]
        Xbatch, Ybatch = next(iter_dataset(Xs=X, Ys=Y, batch_sz=eval_sz))
        # fwd, loss = model(X, target)
        out[split] = eval_model_loss(model=model, X=Xbatch, Ytrue=Ybatch).item()
    model.train()
    return out


def eval_model_loss(model, X, Ytrue):
    return calc_loss(Ypred=model(X=X), Ytrue=Ytrue)


def calc_loss(Ypred, Ytrue):
    """
    Ypred dim: Batch_size x block_sz x vocab_sz
    Ytrue dim: Batch_size x block_sz
    """
    return F.cross_entropy(input=_flatten_y(Ypred), target=Ytrue.view(-1))


def _flatten_y(Y):
    B, T, C = Y.shape
    return Y.view(B * T, C)


@torch.no_grad()
def calc_split_loss(model, data_splits, split):
    model.eval()
    X, Ytrue = data_splits[split]
    loss = model(X=X, target=Ytrue).item()
    model.train()
    return loss


#######################################################################################
#                           MODELS
#######################################################################################


@dataclass(eq=False)
class BigramModel(nn.Module):

    vocab_sz: int
    embed: nn.Embedding = field(init=False)

    def __post_init__(self):
        super().__init__()
        vocab_sz = self.vocab_sz
        self.embed = nn.Embedding(num_embeddings=vocab_sz, embedding_dim=vocab_sz)
        # make predictions less confident at the beginning so that all
        # classes about equal probability.
        self.embed.weight.data = self.embed.weight * 0.01

    def forward(self, X, target=None):
        """Returns the forward pass and NLL loss"""
        out = self.embed(X)
        if target is None:
            return out, None
        else:
            loss = calc_loss(Ypred=out, Ytrue=target)
            return out, loss


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
            *[Block(**block_kwargs) for _ in range(self.num_blocks)],
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
        pos_embed = self.pos_embed_tbl(torch.arange(T, device=DEVICE))  # T x C
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
) -> torch.Tensor:
    """Generates examples from the model. We always begin with a newline character.
    Logic: We always take the prediction of the last time dim, in the block. We add
    the
    """
    # B x T: zero corresponds to the newline character
    # This always contains the current block size of context
    cur_block = torch.zeros(num_examples, block_size, dtype=torch.int, device=DEVICE)
    # This will be appended with the next prediction continuously
    outputs = torch.zeros(num_examples, 1, dtype=torch.int, device=DEVICE)
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
    to_replace_nl=False,
):
    examples = gen_examples(
        model=model,
        max_len=max_len,
        block_size=block_size,
        num_examples=num_examples,
    )
    for ex in examples:
        print("".join(decode(ex, ixtoc=ixtoc, to_replace_nl=to_replace_nl)))
        print("-----------------------------------------------------------")


#######################################################################################
# Load data
#######################################################################################
with open(INPUT_FL, encoding="utf-8") as infile:
    input_txt = infile.read()
vocab = sorted(set(input_txt))
vocab_sz = len(vocab)
ctoix = {char: ix for ix, char in enumerate(vocab)}
ixtoc = {ix: char for ix, char in enumerate(vocab)}
print("Vocab:", vocab)
print(f"Num characters: {len(input_txt):,}\nExample text:")
print()
print(input_txt[:1000])
print("-----------------------------------------------------------")

#######################################################################################
#  Train models
#######################################################################################
# Build training data
print("Building training data matrices")
X, Y = txt_to_data(input_txt=input_txt, ctoix=ctoix, block_size=HYPERPARAMS.block_sz)
print(f"{X.shape=}, {Y.shape=}")
# Train validation split
# we don't want any test data
print("Splitting into train, validation")
data_splits = split_data(X=X, Y=Y, train_frac=HYPERPARAMS.train_frac, to_shuffle=False)

train_nll = ft.partial(calc_split_loss, data_splits=data_splits, split="train")
val_nll = ft.partial(calc_split_loss, data_splits=data_splits, split="val")
eval_loss_fn = ft.partial(eval_loss, data_splits=data_splits, eval_sz=EVAL_SZ)

Xtrain, Ytrain = data_splits["train"]
Xval, Yval = data_splits["val"]
print(f"{Xtrain.shape=}, {Ytrain.shape=}")
print(f"{Xval.shape=}, {Yval.shape=}")
# train_dataset = torch.utils.data.TensorDataset(Xtrain, Ytrain)
# dataset_loader = torch.utils.data.DataLoader(
#     dataset=train_dataset,
#     batch_size=HYPERPARAMS.batch_sz, shuffle=True,
#     num_workers=4,
# )
# Define the model
print("Creating the transformer model")
model = Transformer(
        embed_dim=HYPERPARAMS.embed_dim,
        vocab_sz=vocab_sz,
        block_size=HYPERPARAMS.block_sz,
        num_attn_head=HYPERPARAMS.num_attn_heads,
        mlp_hidden_dim=HYPERPARAMS.mlp_hidden_dim,
        mlp_hidden_layers=HYPERPARAMS.mlp_hidden_layers,
        dropout_frac=HYPERPARAMS.dropout_frac,
        num_blocks=HYPERPARAMS.num_blocks,
    ).to(DEVICE)
model = torch.compile(model=model)
tb_writer.add_graph(model, Xtrain[:4])
print("\nExamples before training the model")
print_examples(
    model=model,
    max_len=1000,
    block_size=HYPERPARAMS.block_sz,
    ixtoc=ixtoc,
    num_examples=2,
)
optimiser = torch.optim.AdamW(params=model.parameters(), lr=HYPERPARAMS.learning_rate)
losses = {"train": [], "val": []}
# Train the model
print("Training the transformer model")
train_model(
    model=model,
    Xtrain=Xtrain,
    Ytrain=Ytrain,
    batch_sz=HYPERPARAMS.batch_sz,
    num_iters=HYPERPARAMS.max_iters,
    optimizer=optimiser,
    loss_calc_freq=EVAL_FREQ,
    losses=losses,
    eval_loss_fn=eval_loss_fn,
)
print("\n\nExamples after training the model")
print_examples(
    model=model,
    max_len=1000,
    block_size=HYPERPARAMS.block_sz,
    ixtoc=ixtoc,
    num_examples=2,
)
el_time = time.time() - st_time
print(f"Time elapsed: {el_time:,}s {el_time//60:,}m")
tb_writer.close()
