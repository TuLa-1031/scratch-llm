# GQA: Grouped Query Attention

This experiment implements a grouped-query attention (GQA) variant of a small GPT-style transformer in PyTorch. The goal is to study how multiple query heads can share a smaller set of key/value heads, reducing the memory footprint of the attention projection while preserving model quality and allowing efficient KV-cache usage.

## What is GQA?

In standard multi-head attention, each query head has its own key and value projection. In GQA, the model uses:

- `n_heads` query heads
- `kv_groups` shared key/value groups
- each group serving multiple query heads

This means the K/V projection size is reduced from:

- `n_heads * head_dim`

to:

- `kv_groups * head_dim`

and the shared K/V vectors are repeated across the corresponding query heads before the attention score computation.

## Implementation notes

The implementation in this folder includes:

- `Attention`: grouped-query attention with optional KV-cache support
- `TransformerBlock`: residual block with attention + feed-forward network
- `LLM`: token embedding, positional embedding, stacked transformer blocks, and output head
- `generate_text_simple_cached`: autoregressive generation loop with and without KV caching

Key logic:

- Query heads are produced as usual with `W_query`
- Key/value projections are produced with fewer heads: `kv_groups * head_dim`
- Shared K/V are expanded back to match the full head count using `repeat_interleave`
- The attention mask enforces causal decoding in generation
- Optional KV cache stores previous key/value states for faster incremental generation


## Example benchmark run

This project was run on Apple Silicon with MPS for a small GPT-style setup. Example output from the current implementation:

```text
Encoded input text: [15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])

Time: 2.07 sec
98 tokens/sec
Current memory allocated: 0.63 GB

NO CACHE:

Time: 4.42 sec
46 tokens/sec
Current memory allocated: 0.63 GB
```

This shows the cached variant is faster because it avoids recomputing all prior attention keys and values at each generation step.

## Summary

This experiment is a practical, minimal version of grouped-query attention. It is designed for learning and exploration rather than production training and is especially useful for understanding:

- how attention is structured in modern LLMs
- how GQA reduces K/V memory cost
- how KV cache accelerates autoregressive decoding
- how to compare cached vs non-cached generation performance