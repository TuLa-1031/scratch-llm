import torch
from torch.utils.checkpoint import checkpoint
from model import trf_block

d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
block = trf_block(
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads,
    max_seq_len=context_length
)

block = torch.compile(block, fullgraph=True)
x = torch.randn((4, context_length, d_model), requires_grad=True)

total_size_bytes = 0
def pack_hook(t):
    if isinstance(t, torch.nn.Parameter):
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    if isinstance(t. torch.nn.Parameter):
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def two_blocks(x):
    x = block(x)
    x = block(x)
    return x

def four_blocks_checkpoint(x):
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks_checkpoint(x)

print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes / (1024**2):.2f} MiB")
