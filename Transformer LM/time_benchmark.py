import torch
import argparse
import timeit


from optimizer import AdamW
from model import *
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)


    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--mode", type=str, required=True)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    model = transformerLanguageModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
    ).to(device)
    model = torch.compile(model, backend="aot_eager")
    
    optimizer = AdamW(model.parameters())

    batch = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(1, args.context_length + 1),
        dtype=torch.int32,
        device=device,
    )
    x = batch[:, :-1]
    y = batch[:, 1:]

    time_run = []

    def run_step():
        if args.mode == "forward":
            with torch.no_grad():
                logits = model(x)
                loss = cross_entroy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                del logits
                del loss

        elif args.mode =="backward":
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = cross_entroy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            loss.backward()
            del logits
            del loss

        elif args.mode == "optimizer":
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = cross_entroy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            loss.backward()
            optimizer.step()
            del logits
            del loss
        torch.mps.synchronize()


    for _ in range(args.w):
        run_step()


    for step in range(10):
        start = timeit.default_timer()
        run_step()
        time_run.append(timeit.default_timer() - start)


    print(f"Time run: {time_run}")
    sum_time = 0
    for t in time_run:
        sum_time += t
    print(f"Mean time: {sum_time/10}")