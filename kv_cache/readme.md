# KV Cache Generation Benchmark

This example compares autoregressive text generation with and without a key-value (KV) cache in a small GPT-style transformer implemented with PyTorch.

The model uses:

- GPT-2 tokenizer vocabulary: `50,257` tokens
- Context length: `1,024` tokens
- Embedding dimension: `768`
- Attention heads: `12`
- Transformer layers: `12`
- Generated tokens: `200`


## Benchmark Results

The benchmark was run on an Apple Silicon Mac using MPS and generated 200 tokens.

### With KV Cache

```text
Time: 1.71 sec
119 tokens/sec
Current memory allocated: 0.68 GB
Max memory allocated: 0.68 GB
```

### Without KV Cache

```text
Time: 3.61 sec
56 tokens/sec
Current memory allocated: 0.68 GB
Max memory allocated: 0.68 GB
```

## Summary

| Mode | Time | Throughput | Current memory | Max memory |
| --- | ---: | ---: | ---: | ---: |
| KV cache | 1.71 sec | 119 tokens/sec | 0.68 GB | 0.68 GB |
| No KV cache | 3.61 sec | 56 tokens/sec | 0.68 GB | 0.68 GB |

For this run, the KV cache generated text approximately **2.1x faster** than recomputing the full context at every step. Memory usage was the same at the displayed precision because the model weights and activation allocations dominate the reported MPS memory usage.

## Files

- `main.py`: Transformer model, KV-cache implementation, generation functions, and benchmark runner.
- `kv_cache.ipynb`: Notebook version of the KV-cache experiment.
