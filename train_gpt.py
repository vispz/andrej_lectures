
import logging
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

import gpt

from dataclasses import dataclass, field, is_dataclass
from tqdm import tqdm



st_time = time.time()


#######################################################################################
# Constants
#######################################################################################
INPUT_FL = "input.txt"
EVAL_FREQ = 500
EVAL_SZ = 3000
# device = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = "cpu"
print(f"Running on device: {DEVICE}")


@dataclass(frozen=True)
class TrainConfig:

    train_frac: float = 0.9
    batch_sz: int = 32
    max_iters: int = 1000


@dataclass(frozen=True)
class TransformerConfig:
    block_sz: int = 8
    # if you update this also update mlp_hidden_dim
    embed_dim: int = 32
    # embed_dim * 4
    mlp_hidden_dim: int = 32 * 4
    num_attn_heads: int = 4
    mlp_hidden_layers: int = 1
    dropout_frac: float = 0.0
    # number of decoder transformer blocks stacked one on top of another
    num_blocks: int = 1


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class HyperParams:
    transformer_cfg: TransformerConfig
    optimizer_cfg: OptimizerConfig
    train_cfg: TrainConfig


def dataclass_to_dict(dc) -> dict:
    out = {}
    for k, v in dc.__dict__.items():
        out[k] = dataclass_to_dict(dc=v) if is_dataclass(v) else v
    return out


HYPERPARAMS = HyperParams(
    train_cfg=TrainConfig(),
    optimizer_cfg=OptimizerConfig(),
    transformer_cfg=TransformerConfig(),
)


#######################################################################################
# SETUP
#######################################################################################
# logging setup
run_name = f'transformer_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
tb_writer = tb.SummaryWriter(log_dir=f"logging/tb/{run_name}")
logging.basicConfig(
    filename=f"logging/logs/{run_name}.log",
    filemode="a",
    format="%(asctime)s %(name)s %(levelname)s::\t %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)
logging.debug(f"{HYPERPARAMS}")


torch.manual_seed(1338)


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
    return torch.tensor(Xs), torch.tensor(Ys)


def encode(txt, ctoix):
    return [ctoix[char] for char in txt]


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
    dataset_loader,
    num_iters,
    optimizer,
    eval_loss_fn,
    losses: dict[str, list],  # train -> [], val -> []
    loss_calc_freq=100,
):
    iters = zip(range(num_iters), dataset_loader)
    for i, (Xi, Yi) in (pbar := tqdm(iters, total=num_iters)):
        loss = eval_model_loss(model=model, X=Xi.to(DEVICE), Ytrue=Yi.to(DEVICE))
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
        out[split] = eval_model_loss(
            model=model, X=Xbatch.to(DEVICE), Ytrue=Ybatch.to(DEVICE)
        ).item()
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
X, Y = txt_to_data(
    input_txt=input_txt, ctoix=ctoix, block_size=HYPERPARAMS.transformer_cfg.block_sz
)
print(f"{X.shape=}, {Y.shape=}")
# Train validation split
# we don't want any test data
print("Splitting into train, validation")
data_splits = split_data(
    X=X, Y=Y, train_frac=HYPERPARAMS.train_cfg.train_frac, to_shuffle=False
)

train_nll = ft.partial(calc_split_loss, data_splits=data_splits, split="train")
val_nll = ft.partial(calc_split_loss, data_splits=data_splits, split="val")
eval_loss_fn = ft.partial(eval_loss, data_splits=data_splits, eval_sz=EVAL_SZ)

Xtrain, Ytrain = data_splits["train"]
Xval, Yval = data_splits["val"]
print(f"{Xtrain.shape=}, {Ytrain.shape=}")
print(f"{Xval.shape=}, {Yval.shape=}")
train_dataset = torch.utils.data.TensorDataset(Xtrain, Ytrain)
dataset_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=HYPERPARAMS.train_cfg.batch_sz,
    shuffle=True,
    num_workers=0,
)
# Define the model
print("Creating the transformer model")
model = gpt.Transformer(
    embed_dim=HYPERPARAMS.transformer_cfg.embed_dim,
    vocab_sz=vocab_sz,
    block_size=HYPERPARAMS.transformer_cfg.block_sz,
    num_attn_head=HYPERPARAMS.transformer_cfg.num_attn_heads,
    mlp_hidden_dim=HYPERPARAMS.transformer_cfg.mlp_hidden_dim,
    mlp_hidden_layers=HYPERPARAMS.transformer_cfg.mlp_hidden_layers,
    dropout_frac=HYPERPARAMS.transformer_cfg.dropout_frac,
    num_blocks=HYPERPARAMS.transformer_cfg.num_blocks,
    device=DEVICE,
).to(DEVICE)
# model = torch.compile(model=model)
tb_writer.add_graph(model, Xtrain[:4].to(DEVICE))
print("\nExamples before training the model")
gpt.print_examples(
    model=model,
    max_len=1000,
    block_size=HYPERPARAMS.transformer_cfg.block_sz,
    ixtoc=ixtoc,
    device=DEVICE,
    num_examples=2,
)
optimiser = torch.optim.AdamW(
    params=model.parameters(), lr=HYPERPARAMS.optimizer_cfg.learning_rate
)
losses = {"train": [], "val": []}
# Train the model
print("Training the transformer model")
train_model(
    model=model,
    dataset_loader=dataset_loader,
    num_iters=HYPERPARAMS.train_cfg.max_iters,
    optimizer=optimiser,
    loss_calc_freq=EVAL_FREQ,
    losses=losses,
    eval_loss_fn=eval_loss_fn,
)
print("\n\nExamples after training the model")
gpt.print_examples(
    model=model,
    max_len=1000,
    block_size=HYPERPARAMS.transformer_cfg.block_sz,
    device=DEVICE,
    ixtoc=ixtoc,
    num_examples=2,
)
el_time = time.time() - st_time
print(f"Time elapsed: {el_time:,}s {el_time//60:,}m")
tb_writer.close()
