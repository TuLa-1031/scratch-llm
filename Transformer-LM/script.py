import torch
import argparse
from transformer_lm import transformerLanguageModel
from optimizer import AdamW
from data import get_batch
from utils import *
from tokenizer import Tokenizer
import pickle
from multiprocessing import Pool  
import numpy as np
import os
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--context_length", type=int, default=256)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--num_heads", type=int, default=16)
parser.add_argument("--d_ff", type=int, default=1344)
parser.add_argument("--rope_theta", type=float, default=10000.0)

parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--min_learning_rate", type=float, default=3e-5)
parser.add_argument("--warmup_iters", type=int, default=200)
parser.add_argument("--max_iters", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))

parser.add_argument("--file_path", required=True)
parser.add_argument("--tokenizer_path", required=True)
parser.add_argument("--val_file_path", default=None)

parser.add_argument("--checkpoint_path", default="checkpoints/tiny")
parser.add_argument("--save_every", type=int, default=500)
parser.add_argument("--log_every", type=int, default=10)

args = parser.parse_args()


def process_chunk(args_tuple):
    file_path, start, end, vocab, merges= args_tuple
    from tokenizer import Tokenizer
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    with open(file_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

    return len(tokenizer.encode(chunk))


def write_chunk(args_tuple):
    file_path, start, end, vocab, merges, output_path, offset, count = args_tuple
    from tokenizer import Tokenizer
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end-start).decode("utf-8", errors="ignore")

    ids = np.asarray(tokenizer.encode(chunk), dtype=np.uint32)
    if len(ids) != count:
        raise ValueError(
            f"Expected {count} tokens, got {len(ids)}"
        )

    out = np.lib.format.open_memmap(output_path, "r+")
    out[offset:offset + count] = ids
    out.flush()
    del out
    return count





if __name__ == "__main__":
    num_processes = 2
    output_path = "data/tiny.npy"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(args.tokenizer_path, "rb") as f:
        vocab, merges = pickle.load(f)

    with open(args.file_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, b"<|endoftext|>"
        )

    chunks = [
        (args.file_path, start, end, vocab, merges)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(processes=num_processes) as pool:
        token_counts = pool.map(process_chunk, chunks)

    offsets = np.cumsum([0] + token_counts[:-1]).tolist()
    total_tokens = sum(token_counts)

    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint32,
        shape=(total_tokens,),
    )
    out.flush()
    del out

    write_tasks = [
        (
            raw_path,
            start,
            end,
            vocab,
            merges,
            output_path,
            offset,
            count,
        )
        for (raw_path, start, end, vocab, merges), offset, count
        in zip(chunks, offsets, token_counts)
    ]

    with Pool(processes=num_processes) as pool:
        pool.map(write_chunk, write_tasks)

    print(f"Tokenized {total_tokens:,} tokens into {output_path}")


    file_path = "data/tiny.npy"
    checkpoint_path = args.checkpoint_path
    os.makedirs(checkpoint_path, exist_ok=True)

    train_data = np.load(output_path, mmap_mode="r")
    assert args.vocab_size == len(vocab), (
        f"Model vocab_size={args.vocab_size}, "
        f"tokenizer vocab_size={len(vocab)}"
    )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"


    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="gilytula",
        # Set the wandb project where this run will be logged.
        project="asignment1",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": args.learning_rate,
            "max_iters": args.max_iters,
        },
    )


    model = transformerLanguageModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.rope_theta
    ).to(device)
    model = torch.compile(model, backend="aot_eager")
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay
    )

    for step in range(args.max_iters):
        lr = lr_cosin_schedule(
            step,
            args.learning_rate,
            args.min_learning_rate,
            args.warmup_iters,
            args.max_iters
        )


        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        logits = model(x)
        loss = cross_entroy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        run.log({
            "train/loss": loss.item(),
            "train/learning_rate": lr,
            "train/step": step + 1,
        }, step=step + 1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        current_step = step + 1
        if current_step % args.save_every == 0 or current_step == args.max_iters:
            print(f"step {step+1}, loss = {loss.item():.4f}, lr={lr:.2e}")
            out = os.path.join(checkpoint_path, f"checkpoint_step{step+1}.pt")
            save_checkpoint(model, optimizer, step, out)

    run.finish()