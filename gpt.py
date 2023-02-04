import functools as ft
import matplotlib.pyplot as plt
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb

from typing import Optional
from dataclasses import dataclass, field
from tqdm import tqdm


torch.manual_seed(1337)


plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["font.size"] = 14



st_time = time.time()
#######################################################################################
# Hyperparams
#######################################################################################
INPUT_FL = "input.txt"
EVAL_FREQ = 100
EVAL_SZ = 5000
# device = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = "cpu"
print(f"Running on device: {DEVICE}")

@dataclass
class HyperParams:
    block_z = 8
    train_frac = 0.9
    val_frac = 0.1
    batch_sz = 32
    max_iters = 3_000
    learning_rate = 1e-2
    embed_dim = 32

HYPERPARAMS = HyperParams()


tb_writer = tb.SummaryWriter(comment=str(HYPERPARAMS))



#######################################################################################
# Text to data
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
# Util function to iterate over a random batch
#######################################################################################

def iter_dataset(Xs, Ys, batch_sz, num=None):
    num = num or torch.inf
    i = 0
    n = Xs.shape[0]
    while i < num:
        i += 1
        batch_ixes = torch.randint(n, (batch_sz,))
        yield Xs[batch_ixes], Ys[batch_ixes]


def split_data(X, Y, train_frac=0.8, val_frac=0.1, to_shuffle=True):
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
        probas, loss = model(X=Xi, target=Yi)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        # Metric tracking
        if i % loss_calc_freq == 0:
            eval_losses = eval_loss_fn(model=model)
            tb_writer.add_scalars(main_tag="loss", tag_scalar_dict=eval_losses)
            loss_msg = ""
            for split, loss in eval_losses.items():
                losses[split].append(loss)
                loss_msg += f"{split} loss: {loss:.4f} "
            pbar.set_description(loss_msg)


def plot_loss(losses, avg_win=1, loss_calc_freq=1):
    plt.figure(figsize=(10, 5))
    for split, split_losses in losses.items():
        n = avg_win * (len(split_losses) // avg_win)
        plt.grid(True)
        plt.plot(
            torch.arange(0, n, avg_win) * loss_calc_freq,
            torch.tensor(split_losses[:n])
            .view(-1, avg_win)
            .mean(dim=1)
            .detach()
            .numpy(),
            label=split,
            linewidth=2,
        )
    plt.title(f"Loss vs iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()


@torch.no_grad()
def eval_loss(model, data_splits, eval_sz):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        X, Y = data_splits[split]
        Xbatch, Ybatch = next(iter_dataset(Xs=X, Ys=Y, batch_sz=eval_sz))
        # fwd, loss = model(X, target)
        out[split] = model(X=Xbatch, target=Ybatch)[1].item()
    model.train()
    return out


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
        self.embed = nn.Embedding(
            num_embeddings=vocab_sz, embedding_dim=vocab_sz
        )
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
        return out


@dataclass(eq=False)
class Transformer(nn.Module):

    vocab_sz: int
    embed_dim: int
    block_size: int
    char_embed: nn.Embedding = field(init=False)
    pos_embed: nn.Embedding = field(init=False)

    def __post_init__(self):
        super().__init__()
        vocab_sz, embed_dim = self.vocab_sz, self.embed_dim
        self.char_embed = nn.Embedding(
            num_embeddings=vocab_sz, embedding_dim=embed_dim
        )
        self.pos_embed = nn.Embedding(
            num_embeddings=self.block_size, embedding_dim=embed_dim
        )
        self.logit_layer = nn.Linear(in_features=embed_dim, out_features=vocab_sz)
        # make predictions less confident at the beginning so that all
        # classes about equal probability.
        self.logit_layer.weight.data = self.logit_layer.weight * 0.01

    def forward(self, X, target=None):
        """Returns the forward pass and NLL loss"""
        B, T = X.shape
        x = self.char_embed(X)  # B x T x C
        out = self.logit_layer(x)  # B x T x V
        if target is None:
            return out, None
        else:
            loss = calc_loss(Ypred=out, Ytrue=target)
            return out, loss

@dataclass(eq=False)
class AttentionHead:

    embed_dim: int
    block_size: int

    def __post_init__(self):
        super().__init__()
        embed_dim, block_size = self.embed_dim, self.block_size
        lin_kwargs = dict(in_features=embed_dim, out_features=block_size)
        self.key = nn.Linear(**lin_kwargs)
        self.query = nn.Linear(**lin_kwargs)
        self.value = nn.Linear(**lin_kwargs)
    


#######################################################################################
# Generate examples from models
#######################################################################################

def gen_examples(
    model: nn.Module,
    max_len: int,
    start_ix_tns: torch.Tensor = None,
    num_examples: int = None,
) -> torch.Tensor:
    """Generates examples from the model.

    Either
    * start_ix_tns will pass in the first token for each word
    * or we use num_examples to arrive at the number of examples and
        we always begin with a newline character.
    """
    assert not (
        start_ix_tns is None and num_examples is None
    ), "Either pass first token or pass num examples"
    # B x T x C
    if start_ix_tns is None:
        start_ix_tns = torch.zeros(num_examples, 1, dtype=torch.int, device=DEVICE)
    examples = start_ix_tns.clone()
    for i in range(max_len):
        # B x C
        logits, loss = model(examples[:, [-1]])
        # B x C
        probas = F.softmax(logits, dim=1)
        # B x 1
        next_ix = torch.multinomial(probas.squeeze(), num_samples=1)
        examples = torch.concat((examples, next_ix), axis=1)
    return examples


def print_examples(
    model: nn.Module,
    start_ix_tns: torch.Tensor,
    max_len: int,
    ixtoc: dict,
    num_examples: int = None,
    to_replace_nl=False,
) -> str:
    examples = gen_examples(
        model=model,
        start_ix_tns=start_ix_tns,
        max_len=max_len,
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
X, Y = txt_to_data(input_txt=input_txt, ctoix=ctoix, block_size=BLOCK_SZ)
print(f"{X.shape=}, {Y.shape=}")
# Train validation split
# we don't want any test data
print("Splitting into train, validation")
data_splits = split_data(
    X=X, Y=Y, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, to_shuffle=False
)

train_nll = ft.partial(calc_split_loss, data_splits=data_splits, split="train")
val_nll = ft.partial(calc_split_loss, data_splits=data_splits, split="val")
eval_loss_fn = ft.partial(eval_loss, data_splits=data_splits, eval_sz=EVAL_SZ)

Xtrain, Ytrain = data_splits["train"]
Xval, Yval = data_splits["val"]
print(f"{Xtrain.shape=}, {Ytrain.shape=}")
print(f"{Xval.shape=}, {Yval.shape=}")
# Define the model
print("Creating the transformer model")
model = Transformer(embed_dim=EMBED_DIM, vocab_sz=vocab_sz).to(DEVICE)
print("\nExamples before training the model")
print_examples(
    model=model,
    max_len=1000,
    # assumes start token is newline char
    start_ix_tns=None,
    ixtoc=ixtoc,
    num_examples=2,
)
optimiser = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
losses = {"train": [], "val": []}
# Train the model
print("Training the transformer model")
train_model(
    model=model,
    Xtrain=Xtrain,
    Ytrain=Ytrain,
    batch_sz=BATCH_SZ,
    num_iters=MAX_ITERS,
    optimizer=optimiser,
    loss_calc_freq=EVAL_FREQ,
    losses=losses,
    eval_loss_fn=eval_loss_fn,
)
print("\n\nExamples after training the model")
print_examples(
    model=model,
    max_len=1000,
    # assumes start token is newline char
    start_ix_tns=None,
    ixtoc=ixtoc,
    num_examples=2,
)
plot_loss(losses, loss_calc_freq=EVAL_FREQ)
el_time = time.time() - st_time
print(f"Time elapsed: {el_time:,}s {el_time//60:,}m")
plt.savefig("train-val-loss.png")