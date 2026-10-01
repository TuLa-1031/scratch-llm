import time

import torch
import torch.nn as nn
from einops import rearrange
import tiktoken

LLM_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "kv_groups": 2,
}


class Attention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        num_heads,
        dropout,
        kv_groups,
        dtype=None,
        qkv_bias=False,
    ):
        super().__init__()
        head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, kv_groups * head_dim, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, kv_groups * head_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_prj = nn.Linear(d_out, d_out, bias=False)

        self.num_heads = num_heads
        self.kv_groups = kv_groups
        self.group_size = num_heads // kv_groups

        #init in the same device with the model but do not have grad
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

        self.cur_pos = 0 # pointer to the current pos of the new token

    def forward(self, x, use_cache=False):
        q = rearrange(self.W_query(x), "b l (h k) -> b h l k", h=self.num_heads)
        k = rearrange(self.W_key(x), "b t (h k) -> b h t k", h=self.kv_groups)
        v = rearrange(self.W_value(x), "b t (h v) -> b h t v", h=self.kv_groups)

        if use_cache:
            if self.k_cache is None:
                self.k_cache, self.v_cache = k, v
            else:
                self.k_cache = torch.cat([self.k_cache, k], dim=2)
                self.v_cache = torch.cat([self.v_cache, v], dim=2)
            k, v = self.k_cache, self.v_cache

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        nQ = q.shape[-2]
        nK = k.shape[-2]

        att_scores = torch.einsum("bhlk, bhtk -> bhlt", [q, k]) / k.shape[-1] ** 0.5

        if use_cache:
            mask_bool = self.mask.bool()[self.cur_pos : self.cur_pos + nQ, :nK]
            self.cur_pos += nQ
        else:
            mask_bool = self.mask.bool()[:nQ, :nK]

        att_scores.masked_fill_(mask_bool, -torch.inf)
        att_scores = torch.softmax(att_scores, dim=-1)
        att_scores = self.dropout(att_scores)

        context_vec = torch.einsum("bhlt, bhtv -> bhlv", [att_scores, v])
        context_vec = self.out_prj(rearrange(context_vec, "b h l v -> b l (h v)"))

        return context_vec

    def reset_cache(self) -> None:
        self.k_cache, self.v_cache = None, None
        self.cur_pos = 0


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = Attention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            dropout=config["drop_rate"],
            num_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"],
            kv_groups=config["kv_groups"],
        )
        self.ff = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            nn.SiLU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )
        self.norm1 = nn.RMSNorm(config["emb_dim"])
        self.norm2 = nn.RMSNorm(config["emb_dim"])
        self.drop = nn.Dropout(config["drop_rate"])

    def forward(self, x, use_cache=False):
        sc = x
        x = self.att(x, use_cache=use_cache)
        x = self.norm1(x)
        x += sc

        sc = x
        x = self.ff(x)
        x = self.norm2(x)
        x += sc
        return x


class LLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop = nn.Dropout(config["drop_rate"])
        self.tranformerBlock = nn.ModuleList([
            TransformerBlock(config) for _ in range(config["n_layers"])
        ])
        self.cur_pos = 0
        self.final_norm = nn.RMSNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        _, seq_len = in_idx.shape
        tok_emb = self.tok_emb(in_idx)
        if use_cache:
            pos_ids = torch.arange(
                self.cur_pos,
                self.cur_pos + seq_len,
                device=in_idx.device,
                dtype=torch.long,
            )
            self.cur_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_emb = self.pos_emb(pos_ids).unsqueeze(0)
        x = tok_emb + pos_emb
        x = self.drop(x)
        for block in self.tranformerBlock:
            x = block(x, use_cache=use_cache)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self):
        for block in self.tranformerBlock:
            block.att.reset_cache()
        self.cur_pos = 0


def generate_text_simple_cached(model, idx, max_new_tokens, context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    torch.manual_seed(123)
    model = LLM(LLM_CONFIG)
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.mps.is_available():
        torch.mps.synchronize()
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200,
    )

    if torch.mps.is_available():
        torch.mps.synchronize()
    total_time = time.time() - start

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0]) / total_time)} tokens/sec")
    if torch.mps.is_available():
        mem_bytes = torch.mps.current_allocated_memory()
        mem_gb = mem_bytes / (1024 ** 3)
        print(f"Current memory allocated: {mem_gb:.2f} GB")

    print("\nNO CACHE:\n")
    if torch.mps.is_available():
        torch.mps.synchronize()
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200,
        use_cache=False,
    )

    if torch.mps.is_available():
        torch.mps.synchronize()
    total_time = time.time() - start

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0]) / total_time)} tokens/sec")
    if torch.mps.is_available():
        mem_bytes = torch.mps.current_allocated_memory()
        mem_gb = mem_bytes / (1024 ** 3)
        print(f"Current memory allocated: {mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
