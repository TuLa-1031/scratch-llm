import torch
from torch import nn
from cs336_basics.utils import scaled_dot_product_attention
import argparse
import timeit
import gc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)

    args = parser.parse_args()


    if torch.mps.is_available():
        device = "mps"
    else: device = "cpu"
        

    #dtype = torch.bfloat16
    warmups = 3
    batch_size = 8
    d_models = [16, 32, 64, 128]
    sequence_lengths = [256, 1024, 4096, 8192, 16384]

    def run(
        d_model: int,
        sequence_length: int,
    ):
        q = k = v = None
        try:
            q = torch.randn((batch_size, sequence_length, d_model), device=device)
            k  = torch.randn((batch_size, sequence_length, d_model), device=device)
            v  = torch.randn((batch_size, sequence_length, d_model), device=device)
            if args.mode =="forward":
                for _ in range(warmups):
                    scaled_dot_product_attention(q, k, v)

                start_forward = timeit.default_timer()
                for _ in range(100):
                    scaled_dot_product_attention(q, k, v)
                    torch.mps.synchronize()
                forward_time = timeit.default_timer() - start_forward
                print(f"Forward: {d_model=}, {sequence_length=}, {forward_time=:.6f}")


            elif args.mode == "backward":
                q.requires_grad_(True)
                k.requires_grad_(True)
                v.requires_grad_(True)

                for _ in range(warmups):
                    out = scaled_dot_product_attention(q,k,v)
                    out.sum().backward()
                    q.grad = None
                    k.grad = None
                    v.grad = None

                torch.mps.synchronize()

                out = scaled_dot_product_attention(q, k, v)
                loss = out.sum()
                torch.mps.synchronize()

                curr_mem = torch.mps.current_allocated_memory()
                driver_mem = torch.mps.driver_allocated_memory()

                


                print(
                    f"Before backward: {d_model=}, "
                    f"seq_length: {sequence_length}, "
                    f"curr mem = {curr_mem / 1024**2:.2f} MiB, "
                    f"driver mem = {driver_mem / 1024**2:.2f} MiB"
                )
                del out
                del loss
                torch.mps.synchronize()
                start_backward = timeit.default_timer()

                for _ in range(100):
                    out = scaled_dot_product_attention(q,k,v)
                    loss = out.sum()
                    loss.backward()

                    q.grad = None
                    k.grad = None
                    v.grad = None
                    del loss
                    del out

                torch.mps.synchronize()
                backward_time = timeit.default_timer() - start_backward

                print(f"Backward: {backward_time=}")

                

            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[{args.mode}]OOM: {d_model=}, {sequence_length=}")
            else:
                raise

        finally:
            del q, k, v
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()


    for d_model in d_models:
        for sequence_length in sequence_lengths:
            run(
                d_model,
                sequence_length,
            )
