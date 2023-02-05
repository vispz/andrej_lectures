import os
import os.path
import sys
import logging
import datetime
import functools as ft
import random
import time
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.utils.tensorboard as tb
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from typing import Dict

import gpt

from dataclasses import dataclass, field, is_dataclass
from tqdm import tqdm


st_time = time.time()


#######################################################################################
# Constants
#######################################################################################
INPUT_FL = "input.txt"
# device = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = "cuda:0"
print(f"Running on device: {DEVICE}")
RUN_ID = f'sml_transformer_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
# Set this to the path to load
# "logging/checkpoints/sml_transformer_2023-02-05_21:14:50/8000.pt"
LOAD_MODEL_CKPT_PATH = None 

@dataclass(frozen=True)
class TrainConfig:

    train_frac: float = 0.9
    batch_sz: int = 64
    max_iters: int = 4_001
    save_every: int = 2000
    checkpoint_folder: str = f"logging/checkpoints/{RUN_ID}/"
    eval_sz: int = 1000
    eval_freq: int = 100
    nproc_workers: int = 0


@dataclass(frozen=True)
class TransformerConfig:
    block_sz: int = 128
    # if you update this also update mlp_hidden_dim
    embed_dim: int = 64
    # embed_dim * 4
    mlp_hidden_dim: int = 64 * 4
    num_attn_heads: int = 4
    mlp_hidden_layers: int = 1
    dropout_frac: float = 0.1
    # number of decoder transformer blocks stacked one on top of another
    num_blocks: int = 2


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4


@dataclass(frozen=True)
class HyperParams:
    transformer_cfg: TransformerConfig
    optimizer_cfg: OptimizerConfig
    train_cfg: TrainConfig


HYPERPARAMS = HyperParams(
    train_cfg=TrainConfig(),
    optimizer_cfg=OptimizerConfig(),
    transformer_cfg=TransformerConfig(),
)

#######################################################################################
# SETUP
#######################################################################################


def setup_logging():
    # logging setup
    for folder in ("logs", "tb", "checkpoints"):
        os.makedirs(f"logging/{folder}/{RUN_ID}/")

    tb_writer = tb.SummaryWriter(log_dir=f"logging/tb/{RUN_ID}")
    logging.basicConfig(
        filename=f"logging/logs/{RUN_ID}.log",
        filemode="a",
        format="%(asctime)s %(name)s %(levelname)s::\t %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"{HYPERPARAMS}")
    return tb_writer


#######################################################################################
# Dataset: Text to data
#######################################################################################


def txt_to_data(input_txt, ctoix, block_size, device):
    Xs, Ys = [], []
    input_len = len(input_txt)
    input_ixes = encode(txt=input_txt, ctoix=ctoix)
    for i in range(0, input_len - block_size, block_size):
        Xs.append(input_ixes[i : i + block_size])
        Ys.append(input_ixes[i + 1 : i + block_size + 1])
    return torch.tensor(Xs).to(device), torch.tensor(Ys).to(device)


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
    # dataloader,
    Xtrain,
    Ytrain,
    num_iters,
    optimizer,
    eval_loss_fn,
    checkpoint_folder,
    save_every,
    device,
    tb_writer,
    batch_sz,
    losses: Dict[str, list],  # train -> [], val -> []
    loss_calc_freq=100,
):
    prev_val_loss = 1e5
    iters = zip(
        range(num_iters),
        iter_dataset(Xs=Xtrain, Ys=Ytrain, batch_sz=batch_sz, num=num_iters),
    )
    for i, (Xi, Yi) in (pbar := tqdm(iters, total=num_iters)):
        loss = eval_model_loss(model=model, X=Xi.to(device), Ytrue=Yi.to(device))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        # Metric tracking
        if i== num_iters-1 or i % loss_calc_freq == 0:
            eval_losses = eval_loss_fn(model=model)
            tb_writer.add_scalars(
                main_tag="loss", tag_scalar_dict=eval_losses, global_step=i
            )
            loss_msg = ""
            for split, loss in eval_losses.items():
                losses[split].append(loss)
                loss_msg += f"{split} loss: {loss:.4f} "
            pbar.set_description(loss_msg)
            if (i == num_iters-1 or i % save_every == 0) and eval_losses["val"] < prev_val_loss:
                _checkpoint(model=model, checkpoint_folder=checkpoint_folder, it=i)
            prev_val_loss = eval_losses["val"]


def _checkpoint(model, checkpoint_folder, it):
    fl = os.path.join(checkpoint_folder, f"{it}.pt")
    msg = f"Iteration {it}: Checkpointing model at {fl}"
    print(msg)
    logging.info(msg)
    torch.save(model.state_dict(), f=fl)


@torch.no_grad()
def eval_loss(model, data_splits, eval_sz, device):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        X, Y = data_splits[split]
        Xbatch, Ybatch = next(iter_dataset(Xs=X, Ys=Y, batch_sz=eval_sz))
        # fwd, loss = model(X, target)
        out[split] = eval_model_loss(
            model=model, X=Xbatch.to(device), Ytrue=Ybatch.to(device)
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
def load_input_file(input_fl):
    with open(input_fl, encoding="utf-8") as infile:
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
    return input_txt, vocab_sz, ctoix, ixtoc


#######################################################################################
#  Preparing dataset and building model utilities
#######################################################################################
def build_train_matrices(input_txt, ctoix, block_size, train_frac, device):
    # Build training data
    print("Building training data matrices")
    X, Y = txt_to_data(
        input_txt=input_txt, ctoix=ctoix, block_size=block_size, device=device
    )
    print(f"{X.shape=}, {Y.shape=}")
    # Train validation split
    # we don't want any test data
    print("Splitting into train, validation")
    data_splits = split_data(X=X, Y=Y, train_frac=train_frac, to_shuffle=False)
    Xtrain, Ytrain = data_splits["train"]
    Xval, Yval = data_splits["val"]
    print(f"{Xtrain.shape=}, {Ytrain.shape=}")
    print(f"{Xval.shape=}, {Yval.shape=}")
    return data_splits


def build_dataloader(Xtrain, Ytrain, batch_sz, num_workers):
    train_dataset = torch.utils.data.TensorDataset(Xtrain, Ytrain)
    return torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_sz,
        num_workers=num_workers,
        pin_memory=True,
        # sampler=DistributedSampler(train_dataset),
    )


def build_model(vocab_sz, transformer_cfg, device):
    # Define the model
    print("Creating the transformer model")
    model = gpt.Transformer(
        embed_dim=transformer_cfg.embed_dim,
        vocab_sz=vocab_sz,
        block_size=transformer_cfg.block_sz,
        num_attn_head=transformer_cfg.num_attn_heads,
        mlp_hidden_dim=transformer_cfg.mlp_hidden_dim,
        mlp_hidden_layers=transformer_cfg.mlp_hidden_layers,
        dropout_frac=transformer_cfg.dropout_frac,
        num_blocks=transformer_cfg.num_blocks,
        device=device,
    ).to(device)
    # model = torch.compile(model=model)
    return model


#######################################################################################
#  Main logic function
#######################################################################################


def main(
    input_fl=INPUT_FL,
    device=DEVICE,
    hyperparams=HYPERPARAMS,
    load_model_ckpt_path=LOAD_MODEL_CKPT_PATH,
):
    torch.manual_seed(1338)
    tb_writer = setup_logging()
    input_txt, vocab_sz, ctoix, ixtoc = load_input_file(input_fl=input_fl)
    train_cfg, transformer_cfg = hyperparams.train_cfg, hyperparams.transformer_cfg
    data_splits = build_train_matrices(
        input_txt=input_txt,
        ctoix=ctoix,
        block_size=transformer_cfg.block_sz,
        train_frac=train_cfg.train_frac,
        device=device,
    )
    Xtrain, Ytrain = data_splits["train"]
    # dataloader = build_dataloader(
    #     Xtrain=Xtrain,
    #     Ytrain=Ytrain,
    #     batch_sz=train_cfg.batch_sz,
    #     num_workers=train_cfg.nproc_workers,
    # )
    model = build_model(
        vocab_sz=vocab_sz,
        transformer_cfg=transformer_cfg,
        device=device,
    )
    if load_model_ckpt_path is not None:
        model.load_state_dict(torch.load(load_model_ckpt_path))
    tb_writer.add_graph(model, Xtrain[:4].to(device))
    print("\nExamples before training the model")
    gpt.print_examples(
        model=model,
        max_len=1000,
        block_size=transformer_cfg.block_sz,
        ixtoc=ixtoc,
        device=device,
        num_examples=2,
    )
    optimiser = torch.optim.AdamW(
        params=model.parameters(), lr=hyperparams.optimizer_cfg.learning_rate
    )
    # Train the model
    print("Training the transformer model")
    train_model(
        model=model,
        Xtrain=Xtrain,
        Ytrain=Ytrain,
        # dataloader=dataloader,
        num_iters=train_cfg.max_iters,
        optimizer=optimiser,
        loss_calc_freq=train_cfg.eval_freq,
        losses={"train": [], "val": []},  # not used, see tensorboard instead
        eval_loss_fn=ft.partial(
            eval_loss, data_splits=data_splits, eval_sz=train_cfg.eval_sz, device=device
        ),
        save_every=train_cfg.save_every,
        checkpoint_folder=train_cfg.checkpoint_folder,
        device=device,
        tb_writer=tb_writer,
        batch_sz=train_cfg.batch_sz,
    )
    print("\n\nExamples after training the model")
    gpt.print_examples(
        model=model,
        max_len=1000,
        block_size=transformer_cfg.block_sz,
        device=device,
        ixtoc=ixtoc,
        num_examples=2,
    )
    el_time = time.time() - st_time
    print(f"Time elapsed: {el_time:,}s {el_time//60:,}m")
    tb_writer.close()


if __name__ == "__main__":
    main()
