import regex as re
import pickle
from typing import Iterable, Iterator
from functools import lru_cache


class Tokenizer():
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.special_tokens = special_tokens
        self.vocab_dict = {token: i for i, token in vocab.items()}
        self.vocab = vocab
        self.merges_dict = {
            pair : rank for rank, pair in enumerate(merges)
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, 'rb') as file:
            vocab = pickle.load(file)
        with open(merges_filepath, 'rb') as file:
            merges = pickle.load(file)
        return cls(vocab, merges, special_tokens)

    @lru_cache(maxsize = 10000)
    def to_byte_tuple(self, word):
            return tuple(bytes([b]) for b in word.encode('utf-8'))
    
    def get_stats(self, ids):
        counts = {} 
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i+=1
        return newids

    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        ids = []

        if self.special_tokens is not None:
            pattern = "|".join(re.escape(token)
                            for token in sorted(self.special_tokens, key=len, reverse=True))
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]


        for part in parts:
            if not part:
                continue

            if self.special_tokens is not None and part in self.special_tokens:
                ids.append(self.vocab_dict[part.encode('utf-8')])
            else:
                words = re.finditer(PAT, part)
                for match in words:
                    word = match.group()
                    byte_tuple = self.to_byte_tuple(word)
                    ids_w = list(byte_tuple)
                    while (len(ids_w) >= 2):
                        stats = self.get_stats(ids_w)
                        pair = min(stats, key=lambda p: self.merges_dict.get(p, float("inf")))
                        if pair not in self.merges_dict:
                            break
                        idx = self.merges_dict[pair]
                        ids_w = self.merge(ids_w, pair, pair[0]+pair[1])
                    ids.extend(self.vocab_dict[token] for token in ids_w)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode('utf-8', errors='replace')
        return text

    