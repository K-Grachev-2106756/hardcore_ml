import json
from collections import defaultdict

import numpy as np


class Tokenizer:

    def __init__(self, merges_pth: str = None):
        merges = []
        if merges_pth:
            with open(merges_pth, "r", encoding="utf-8") as f:
                merges = json.load(f)
        self._init_vocab(merges)


    def _init_vocab(self, merges: list):
        self.merges = merges
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (a, b), token_id in merges:
            self.vocab[token_id] = self.vocab[a] + self.vocab[b]
    

    def _insert_token(tokens: list, token_pair: tuple, token_id: int):
        i, new_tokens = 0, []
        while i < len(tokens):
            if (i + 1 < len(tokens)) and (tokens[i] == token_pair[0]) and (tokens[i + 1] == token_pair[1]):
                new_tokens.append(token_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        
        return new_tokens
    

    def _calc_corpus_entropy(tokens: list):
        counts = defaultdict(int)  
        for token in tokens:
            counts[token] += 1
        probs = np.array(list(counts.values())) / len(tokens)
        
        return len(tokens) * -(np.log2(probs) * probs).sum()
    

    def encode(self, text: str):
        tokens = list(text.encode("utf-8"))
        for merge_pair, token_id in self.merges:
            tokens = Tokenizer._insert_token(tokens, merge_pair, token_id)
        
        return tokens


    def decode(self, tokens: list):
        byte_str = b"".join(self.vocab[token] for token in tokens)
        return byte_str.decode("utf-8", errors="replace")
    

    def save(self, dest_pth: str):
        with open(dest_pth, "w", encoding="utf-8") as f:
            json.dump(self.merges, f, indent=4, ensure_ascii=False)


    def fit(
            self,
            data: str,
            min_entropy_decrease_rate: float = -0.0005, 
            extention_step: int = 8, 
            extention_max: int = 2 ** 14,
        ):

        tokens = list(data.encode("utf-8"))
        prev_entropy = Tokenizer._calc_corpus_entropy(tokens)
        
        merges = []
        for _ in range(extention_max // extention_step):
            for _ in range(extention_step):
                new_token_id = len(merges) + 256

                counts = defaultdict(int)  # Поиск наиболее часто встречающейся пары токенов
                for a, b in zip(tokens, tokens[1:]):
                    p = (a, b)
                    counts[p] += 1
                token_pair = max(counts, key=counts.get)

                merges.append((token_pair, new_token_id))  # Сохранение информации о новой паре

                tokens = Tokenizer._insert_token(tokens, token_pair, new_token_id)  # Обновление последовательности токенов

            entropy = Tokenizer._calc_corpus_entropy(tokens)  # Подсчет общей энтропии для поиска оптимального размера словаря

            if entropy / prev_entropy - 1 > min_entropy_decrease_rate:
                break

            prev_entropy = entropy

        self._init_vocab(merges)