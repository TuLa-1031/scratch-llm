# GQA Comparison Benchmark

This folder implements a minimal grouped-query attention (GQA) transformer and compares it against a standard no-GQA baseline imported from the KV-cache experiment.

The goal is to study the effect of reducing the number of key/value heads while keeping the same number of query heads, and to benchmark generation speed with and without KV-cache reuse.

## What this project compares

The script builds two models:

- `GQA`: uses grouped query attention with `kv_groups=2`
- `No GQA`: uses the standard multi-head attention implementation from `kv_cache.main.LLM`

Both models are initialized with the same base configuration, except that the GQA model uses fewer K/V projections and repeats the shared K/V tensors across query heads.

## Model setup

The key configuration is:

- vocab size: `50257`
- context length: `1024`
- embedding dim: `768`
- attention heads: `12`
- transformer layers: `12`
- dropout: `0.1`
- GQA groups: `2`

In GQA, the attention projection is reduced from:

- `n_heads * head_dim` for K/V

to:

- `kv_groups * head_dim`

and the grouped K/V states are repeated to match the full query-head count before attention is computed.

## Files in this folder

- `main.py`: GQA implementation, baseline import, generation loop, and benchmark runner
- `gqa.ipynb`: notebook version of the same experiment

## Run it

From the repository root:

```bash
python GQA/main.py
```

If you are inside the project environment, this will run both models on the same prompt and print timing and throughput for:

1. GQA with KV cache
2. GQA without KV cache
3. No-GQA baseline with KV cache
4. No-GQA baseline without KV cache

## Benchmark result from the current run

The script uses the GPT-2 tokenizer and generates from the prompt:

```text
Hello, I am
```

This run produced the following timings on the current machine:

```text
Encoded input text: [15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])

GQA:

Time: 1.39 sec
146 tokens/sec
Current memory allocated: 1.29 GB

NO CACHE:

Time: 3.84 sec
53 tokens/sec
Current memory allocated: 1.29 GB

No GQA:

Time: 1.59 sec
128 tokens/sec
Current memory allocated: 1.31 GB

NO CACHE:

Time: 3.23 sec
63 tokens/sec
```

This confirms the expected pattern: the cached decoding path is substantially faster than recomputing the entire context at each step, while the GQA model remains competitive with the no-GQA baseline in this minimal setup.

## Notes

This is a compact educational implementation intended for understanding how GQA changes the attention mechanism and how KV caching affects autoregressive decoding. It is not meant to be a production-scale model or training setup.
