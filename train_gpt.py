"""Train nanoGPT

Usage::

    # train mode
    python train_gpt.py
    # prediction mode: Set the model at :data:`LOAD_MODEL_CKPT_PATH`
    python train_gpt.py predict
"""
import os
import os.path
import sys
import logging
import datetime
import functools as ft
import time
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.utils.tensorboard as tb
from typing import Dict

import gpt

from dataclasses import dataclass
from tqdm import tqdm


st_time = time.time()


#######################################################################################
# Constants
#######################################################################################
INPUT_FL = "input.txt"
# device = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {DEVICE}")
RUN_ID = f'sml_transformer_{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}'
# Set this to the path to load
# "logging/checkpoints/sml_transformer_2023-02-06_00_23_38/4500.pt"
LOAD_MODEL_CKPT_PATH = "logging/checkpoints/final_model/6900.pt"
START_IT = 4800


@dataclass(frozen=True)
class TrainConfig:

    train_frac: float = 0.9
    batch_sz: int = 256
    max_iters: int = 5001
    save_every: int = 300
    checkpoint_folder: str = f"logging/checkpoints/{RUN_ID}/"
    eval_sz: int = 2000
    eval_freq: int = 100
    nproc_workers: int = 0


@dataclass(frozen=True)
class TransformerConfig:
    block_sz: int = 256
    # if you update this also update mlp_hidden_dim
    embed_dim: int = 384
    # embed_dim * 4
    mlp_hidden_dim: int = 384 * 4
    num_attn_heads: int = 6
    dropout_frac: float = 0.2
    # number of decoder transformer blocks stacked one on top of another
    num_blocks: int = 6


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
    for folder in ("tb", "checkpoints"):
        os.makedirs(f"logging/{folder}/{RUN_ID}/")
    os.makedirs(f"logging/logs/", exist_ok=True)
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
# Training loop
#######################################################################################


def train_model(
    model,
    get_batch_fn,
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
    for i in (pbar := tqdm(range(num_iters))):
        Xi, Yi = get_batch_fn(split="train", batch_sz=batch_sz)
        loss = eval_model_loss(model=model, X=Xi.to(device), Ytrue=Yi.to(device))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        # Metric tracking
        if not ((i == num_iters - 1) or (i % loss_calc_freq == 0)):
            continue
        eval_losses = eval_loss_fn(model=model)
        tb_writer.add_scalars(
            main_tag="loss", tag_scalar_dict=eval_losses, global_step=i
        )
        loss_msg = ""
        for split, loss in eval_losses.items():
            losses[split].append(loss)
            loss_msg += f"{split} loss: {loss:.4f} "
        pbar.set_description(loss_msg)
        if ((i == num_iters - 1) or (i % save_every == 0)) and (
            eval_losses["val"] < prev_val_loss
        ):
            _checkpoint(
                model=model, checkpoint_folder=checkpoint_folder, it=i + START_IT
            )
        prev_val_loss = eval_losses["val"]


def _checkpoint(model, checkpoint_folder, it):
    fl = os.path.join(checkpoint_folder, f"{it}.pt")
    msg = f"Iteration {it}: Checkpointing model at {fl}"
    print(msg)
    logging.info(msg)
    torch.save(model.state_dict(), f=fl)


@torch.no_grad()
def eval_loss(model, get_batch_fn, eval_sz):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        Xbatch, Ybatch = get_batch_fn(split=split, batch_sz=eval_sz)
        # fwd, loss = model(X, target)
        out[split] = eval_model_loss(model=model, X=Xbatch, Ytrue=Ybatch).item()
    model.train()
    return out


def eval_model_loss(model, X, Ytrue):
    return calc_loss(Ypred=model(X=X), Ytrue=Ytrue)


def calc_loss(Ypred, Ytrue):
    """
    Ypred dim: Batch_sz x block_sz x vocab_sz
    Ytrue dim: Batch_sz x block_sz
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
def load_input_file(input_fl, train_frac):
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
    data = torch.tensor(encode(txt=input_txt, ctoix=ctoix), dtype=torch.long)
    n = int(train_frac * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_sz, ixtoc


def encode(txt, ctoix):
    return [ctoix[char] for char in txt]


def get_batch_helper(split, batch_sz, train_data, val_data, block_sz, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_sz, (batch_sz,))
    x = torch.stack([data[i : i + block_sz] for i in ix])
    y = torch.stack([data[i + 1 : i + block_sz + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def build_model(vocab_sz, transformer_cfg, device):
    # Define the model
    print("Creating the transformer model")
    model = gpt.Transformer(
        embed_dim=transformer_cfg.embed_dim,
        vocab_sz=vocab_sz,
        block_size=transformer_cfg.block_sz,
        num_attn_head=transformer_cfg.num_attn_heads,
        mlp_hidden_dim=transformer_cfg.mlp_hidden_dim,
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
    train_cfg, transformer_cfg = hyperparams.train_cfg, hyperparams.transformer_cfg
    train_data, val_data, vocab_sz, ixtoc = load_input_file(
        input_fl=input_fl, train_frac=train_cfg.train_frac
    )
    get_batch_fn = ft.partial(
        get_batch_helper,
        train_data=train_data,
        val_data=val_data,
        block_sz=transformer_cfg.block_sz,
        device=device,
    )
    model = build_model(
        vocab_sz=vocab_sz,
        transformer_cfg=transformer_cfg,
        device=device,
    )
    for nm, p in model.named_parameters():
        print(nm, p.shape)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    if load_model_ckpt_path is not None:
        print(f"Loading model from checkpoint: {load_model_ckpt_path}")
        model.load_state_dict(torch.load(load_model_ckpt_path, map_location=torch.device('cpu')))
    else:
        print("No model checkpoint passed. Training from scratch.")
    xbatch, _ = get_batch_fn(split="train", batch_sz=4)
    tb_writer.add_graph(model, xbatch)
    print("\nExamples before training the model")
    gpt.print_examples(
        model=model,
        max_len=1000,
        block_size=transformer_cfg.block_sz,
        ixtoc=ixtoc,
        device=device,
        num_examples=2,
    )
    if sys.argv[1].startswith("predict"):
        "In prediction mode. Exiting!"
        return 
    optimiser = torch.optim.AdamW(
        params=model.parameters(), lr=hyperparams.optimizer_cfg.learning_rate
    )
    # Train the model
    print("Training the transformer model")
    train_model(
        model=model,
        get_batch_fn=get_batch_fn,
        num_iters=train_cfg.max_iters,
        optimizer=optimiser,
        loss_calc_freq=train_cfg.eval_freq,
        losses={"train": [], "val": []},  # not used, see tensorboard instead
        eval_loss_fn=ft.partial(
            eval_loss, get_batch_fn=get_batch_fn, eval_sz=train_cfg.eval_sz
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
