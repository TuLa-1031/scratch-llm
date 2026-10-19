import pickle
import regex as re
import os
from functools import lru_cache
import numpy as np

from typing import BinaryIO
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor

import psutil
import time
import cProfile
import pstats
import resource
import json
from datetime import datetime


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunk: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunk

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previou index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunk + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096 # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_positionn = chunk_boundaries[bi]
        file.seek(initial_positionn) # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size) # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_positionn + found_at
                break
            initial_positionn += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


@lru_cache(maxsize = 100000)
def to_byte_tuple(
    word: str
) -> tuple[bytes, ...]:
    return tuple(bytes([b]) for b in word.encode('utf-8'))

def get_freq_table(
    text: str
) -> dict[tuple[bytes, ...], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    freq_table: dict[tuple[bytes, ...], int] = {}
    words = re.finditer(PAT, text)
    for match in words:
        word = match.group()
        byte_tuple = to_byte_tuple(word)
        freq_table[byte_tuple] = freq_table.get(byte_tuple, 0) + 1
    return freq_table

def _build_index(
    words: list[tuple[tuple[bytes, ...], int]],
) -> tuple[
        defaultdict[tuple[bytes, bytes], int],
        defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
    ]:
    pair_counts = defaultdict(int) # {bytes, bytes} -> int
    pair_to_words = defaultdict(set) # {bytes, bytes} -> set
    for word_id, (tokens, freq) in enumerate(words):
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_id)
    return pair_counts, pair_to_words


def _merge(
    tokens,
    pair,
    new_token
) -> tuple[bytes, ...]:
    res = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1      \
        and tokens[i] == pair[0]    \
        and tokens[i+1] == pair[1]:
            res.append(new_token)
            i += 2
        else:
            res.append(tokens[i])
            i += 1
    return res

def merges(
    freq_table: dict[tuple[bytes, ...], int],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[
        dict[tuple[bytes, ...], int],
        dict[int, bytes],
        list[tuple[bytes, bytes]],
    ]:
    offset = len(special_tokens)
    num_merges = vocab_size - 256 - offset
    merges = []
    vocab = {}
    for i, special_tokens in enumerate(special_tokens):
        vocab[i] = special_tokens.encode('utf-8')
    vocab.update({
        idx + offset: bytes([idx]) for idx in range(256)
    })
    words = list(freq_table.items())

    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    pair_counts, pair_to_words = _build_index(words)

    for merged_idx in range(num_merges):
        if not pair_counts: break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        affected_words = list(pair_to_words[best_pair])

        for word_id in affected_words:
            tokens, freq = words[word_id]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pair_counts[pair] -= freq
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                pair_to_words[pair].discard(word_id)
                if not pair_to_words[pair]:
                    del pair_to_words[pair]

            new_tokens = tuple(_merge(tokens, best_pair, merged_token))
            words[word_id] = (new_tokens, freq)

            for i in range(len(new_tokens) - 1):
                pair = (new_tokens[i], new_tokens[i+1])
                pair_counts[pair] += freq
                pair_to_words[pair].add(word_id)


        vocab[offset + 256 + merged_idx] = merged_token

    new_freq_table = {}
    for tokens, freq in words:
        new_freq_table[tokens] = (new_freq_table.get(tokens, 0) + freq)
    return new_freq_table, vocab, merges


def split(
    text: str,
    special_tokens: list[str],
) -> tuple[str, ...]:
    if not special_tokens: return [text]
    delimter = "|".join(
        re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
    )
    chunks = re.split(delimter, text)
    return chunks

def process_chunk(
    args
) -> dict[dict[tuple[bytes, ...], int]]:
    file_path, start, end, special_tokens = args
    with open(file_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode('utf-8', errors='ignore',)
    local_freq = defaultdict(int)
    for sub_chunk in split(text, special_tokens):
        for token, count in get_freq_table(sub_chunk).items():
            local_freq[token] += count
    return dict(local_freq)


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    vocab_size = 3000
    num_processes = os.cpu_count()
    file_path = "data/TinyStoriesV2-GPT4-train.txt"
    file_size = os.path.getsize(file_path)


    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024**2
    start = time.perf_counter()

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = [
        (file_path, start, end, special_tokens) 
        for start, end in zip(
                            boundaries[:-1], boundaries[1:]
                        ) if start < end
    ]

    profile_pretokenizer = cProfile.Profile()
    profile_pretokenizer.enable()
    start_pre = time.perf_counter()

    global_freq_table = defaultdict(int)

    with ProcessPoolExecutor(
        max_workers = num_processes
    ) as executor:
        for local_freq in executor.map(process_chunk, tasks):
            for token, count in local_freq.items():
                global_freq_table[token] += count

    global_freq_table = dict(global_freq_table)

    pretokenize_time = time.perf_counter() - start_pre
    profile_pretokenizer.disable()

    profile_merge = cProfile.Profile()
    profile_merge.enable()
    start_merge = time.perf_counter()

    _, vocab, merged_list = merges(
        global_freq_table, vocab_size, special_tokens
    )
    merge_time = time.perf_counter() - start_merge
    profile_merge.disable()

    train_time = time.perf_counter() - start
    mem_after = process.memory_info().rss / 1024**2

    with open("output/tokenizer.pkl", "wb") as f:
        pickle.dump((vocab, merged_list), f, protocol=pickle.HIGHEST_PROTOCOL)

    pstats.Stats(profile_pretokenizer).sort_stats("cumtime").print_stats(15)
    pstats.Stats(profile_merge).sort_stats("cumtime").print_stats(15)

    print("\n=== Performance ===")
    print(f"File size:             {file_size/1024**2:.2f} MB")
    print(f"Total time:             {train_time:.2f} s")
    print(f"Pre-tokenization time:  {pretokenize_time:.2f} s")
    print(f"Merge training time:    {merge_time:.2f} s")

    if train_time > 0:
        print(f"Throughput:             {file_size / train_time/1024**2:.2f} MB/s")

    if pretokenize_time > 0:
        print(f"Pretokenize throughput: {file_size / pretokenize_time / 1024**2:.2f} MB/s")

    print("\n=== Memory ===")
    print(f"Current RSS: {process.memory_info().rss / 1024**2:.2f} MB")
    print(f"Peak RSS (self.ru_maxrss): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.2f} MB")

    token_lengths = np.array(
        [len(token) for token in vocab.values()]
    )

    print("\n=== Token Length Statistics (bytes) ===")
    print(f"Min:    {token_lengths.min()}")
    print(f"Mean:   {token_lengths.mean():.2f}")
    print(f"Median: {np.median(token_lengths):.2f}")
    print(f"P90:    {np.percentile(token_lengths, 90):.2f}")
    print(f"P95:    {np.percentile(token_lengths, 95):.2f}")
    print(f"P99:    {np.percentile(token_lengths, 99):.2f}")
    print(f"Max:    {token_lengths.max()}")

    length_buckets = Counter(
        "1 byte" if n == 1 else
        "2-4 bytes" if n <= 4 else
        "5-8 bytes" if n <= 8 else
        "9-16 bytes" if n <= 16 else
        ">16 bytes"
        for n in token_lengths
    )

    print("\n=== Token Length Distribution ===")
    for bucket, count in length_buckets.items():
        print(f"{bucket:12s}: {count:6d} ({count / len(vocab) * 100:.2f}%)")


    print("\n=== Correctness ===")
    print(f"Requested vocab size: {vocab_size}")
    print(f"Actual vocab size:    {len(vocab)}")
    print(f"Requested merges:     {vocab_size - 256 - len(special_tokens)}")
    print(f"Actual merges:        {len(merged_list)}")

    print(f"All byte tokens exist: {all(bytes([i]) in vocab.values() for i in range(256))}")
    print(f"Unique vocab IDs:      {len(vocab) == len(set(vocab.keys()))}")
    print(f"Unique vocab values:   {len(vocab.values()) == len(set(vocab.values()))}")

    stats = {
        "timestamp": datetime.now().isoformat(),
        "file": file_path,
        "file_size_mb": file_size / 1024**2,
        "num_processes": num_processes,
        "num_chunks": len(boundaries) - 1,
        "vocab_size": len(vocab),
        "num_merges": len(merged_list),
        "unique_pre_token_sequences": len(global_freq_table),
        "pretokenize_time_s": pretokenize_time,
        "merge_time_s": merge_time,
        "total_time_s": train_time,
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
    }

    with open("output/train_stats.json", "w") as f:
        json.dump(stats, f, indent=2)