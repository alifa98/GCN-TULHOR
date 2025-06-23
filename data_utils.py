import typing
import pandas as pd
import numpy as np
from torchtext.vocab import vocab
from collections import Counter
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
import torch
import h3
from torch.utils.data import Dataset

class BertSimplePreprocessor(Dataset):
    CLS = "[CLS]"
    PAD = "[PAD]"
    MASK = "[MASK]"

    MASK_PERCENTAGE = 0.4
    MASKED_INDICES_COLUMN = "masked_indices"
    TARGET_COLUMN = "indices"
    TOKEN_MASK_COLUMN = "token_mask"
    LABEL_COLUMN = "user_id"
    OPTIMAL_LENGTH_PERCENTILE = 100

    def __init__(self, raw_df, should_include_text=False):
        self.ds = raw_df.copy()
        self.tokenizer = get_tokenizer(None)
        self.counter = Counter()
        self.vocab_hex = None
        self.optimal_sentence_length = None
        self.should_include_text = should_include_text

        self.columns = (
            [
                "masked_sentence",
                self.MASKED_INDICES_COLUMN,
                "sentence",
                self.TARGET_COLUMN,
                self.TOKEN_MASK_COLUMN,
            ]
            if should_include_text
            else [
                self.MASKED_INDICES_COLUMN,
                self.TARGET_COLUMN,
                self.TOKEN_MASK_COLUMN,
            ]
        )

        self.df = self._prepare_dataset()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        inp = torch.tensor(item[self.MASKED_INDICES_COLUMN]).long()
        token_mask = torch.tensor(item[self.TOKEN_MASK_COLUMN]).bool()
        mask_target = torch.tensor(item[self.TARGET_COLUMN]).long()
        mask_target = mask_target.masked_fill(token_mask, 0)
        attention_mask = (inp == self.vocab_hex[self.PAD]).unsqueeze(0)
        return (
            inp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            attention_mask.to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ),
            token_mask.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            mask_target.to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ),
        )

    def _prepare_dataset(self):
        exp = []
        exp_lens = []

        for _, row in self.ds.iterrows():
            self.counter.update(row["higher_order_trajectory"].split())
            exp.append(row["higher_order_trajectory"])
            exp_lens.append(len(row["higher_order_trajectory"].split()))

        self.optimal_sentence_length = int(
            np.percentile(exp_lens, self.OPTIMAL_LENGTH_PERCENTILE)
        )
        self._fill_vocab()

        processed_rows = [self._create_item(self.tokenizer(seq)) for seq in tqdm(exp)]
        df = pd.DataFrame(processed_rows, columns=self.columns)
        self._find_neighbors()

        df[self.LABEL_COLUMN] = self.ds[self.LABEL_COLUMN]
        return df

    def _fill_vocab(self):
        self.vocab_hex = vocab(self.counter, min_freq=1)
        self.vocab_hex.insert_token(self.CLS, 0)
        self.vocab_hex.insert_token(self.PAD, 1)
        self.vocab_hex.insert_token(self.MASK, 2)
        self.vocab_hex.set_default_index(3)

    def _create_item(self, tokens):
        masked, mask_flags = self._preprocess_sentence(tokens.copy())
        masked_indices = self.vocab_hex.lookup_indices(masked)

        original, _ = self._preprocess_sentence(tokens.copy(), should_mask=False)
        original_indices = self.vocab_hex.lookup_indices(original)

        return (
            (masked, masked_indices, original, original_indices, mask_flags)
            if self.should_include_text
            else (masked_indices, original_indices, mask_flags)
        )

    def _preprocess_sentence(self, tokens, should_mask=True):
        if should_mask:
            tokens, mask_flags = self._mask_sentence(tokens)
        else:
            mask_flags = [True] * len(tokens)
        return self._pad_sentence([self.CLS] + tokens, mask_flags)

    def _mask_sentence(self, tokens):
        length = len(tokens)
        mask_flags = [True] * max(length, self.optimal_sentence_length)
        num_to_mask = round(length * self.MASK_PERCENTAGE)

        for _ in range(num_to_mask):
            idx = np.random.randint(0, length)
            if np.random.rand() < 0.8:
                tokens[idx] = self.MASK
            else:
                tokens[idx] = self.vocab_hex.lookup_token(
                    np.random.randint(0, len(self.vocab_hex) - 6)
                )
            mask_flags[idx] = False
        return tokens, mask_flags

    def _pad_sentence(self, sentence: typing.List[str], inverse_token_mask: typing.List[bool] = None):
        len_s = len(sentence)
        target_len = self.optimal_sentence_length

        if inverse_token_mask is not None:
            # Align mask length with sentence after adding [CLS]
            inverse_token_mask = [True] + inverse_token_mask
            len_m = len(inverse_token_mask)

            # Padding logic for mask
            if len_m >= target_len:
                inverse_token_mask = inverse_token_mask[:target_len]
            else:
                inverse_token_mask = inverse_token_mask + [True] * (target_len - len_m)
        else:
            inverse_token_mask = [True] * target_len

        # Padding logic for sentence
        if len_s >= target_len:
            sentence = sentence[:target_len]
        else:
            sentence = sentence + [self.PAD] * (target_len - len_s)

        return sentence, inverse_token_mask


    def _find_neighbors(self):
        self.ds["splited"] = self.ds["higher_order_trajectory"].str.split()
        unique_hexes = set(hex_id for row in self.ds["splited"] for hex_id in row)

        self.neighbor_dict = {hex_id: h3.grid_ring(hex_id) for hex_id in unique_hexes}
        self.pair_edges = []
        self.pair_edges_mapped = []

        for origin, neighbors in self.neighbor_dict.items():
            for neigh in neighbors:
                self.pair_edges.append([origin, neigh])
                if neigh not in self.vocab_hex:
                    self.vocab_hex.insert_token(neigh, len(self.vocab_hex))

        for origin, neigh in self.pair_edges:
            self.pair_edges_mapped.append(
                [
                    self.vocab_hex.lookup_indices([origin])[0],
                    self.vocab_hex.lookup_indices([neigh])[0],
                ]
            )


class BertSimpleTULPreprocessor(Dataset):
    CLS = "[CLS]"
    PAD = "[PAD]"
    HEX_INDICES_COLUMN = "hex_indices"
    USER_ID_COLUMN = "User_id"
    OPTIMAL_LENGTH_PERCENTILE = 100

    def __init__(self, raw_df, hex_vocab, user_vocab=None):
        self.ds = raw_df.copy()
        self.vocab = hex_vocab
        self.user_vocab = user_vocab
        self.tokenizer = get_tokenizer(None)
        self.user_counter = Counter()
        self.columns = [self.HEX_INDICES_COLUMN, self.USER_ID_COLUMN]
        self.df = self._prepare_dataset()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        inp = torch.tensor(item[self.HEX_INDICES_COLUMN]).long()
        attention_mask = (inp == self.vocab[self.PAD]).unsqueeze(0)
        label = torch.tensor([item[self.USER_ID_COLUMN]]).long()
        return (
            inp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            attention_mask.to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ),
            label.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        )

    def _prepare_dataset(self):
        exp = []
        exp_lens = []
        user_ids = []

        for _, row in self.ds.iterrows():
            exp.append(row["higher_order_trajectory"])
            exp_lens.append(len(row["higher_order_trajectory"].split()))
            user_ids.append(str(row["user_id"]))
            self.user_counter.update([str(row["user_id"])])

        self.optimal_sentence_length = int(
            np.percentile(exp_lens, self.OPTIMAL_LENGTH_PERCENTILE)
        )

        if self.user_vocab is None:
            self.user_vocab = vocab(self.user_counter, min_freq=1)

        processed = [
            self._create_item(self.tokenizer(seq), uid)
            for seq, uid in zip(exp, user_ids)
        ]
        return pd.DataFrame(processed, columns=self.columns)

    def _create_item(self, tokens, user_id):
        padded_tokens = self._pad_sentence([self.CLS] + tokens)
        token_indices = self.vocab.lookup_indices(padded_tokens)
        user_index = self.user_vocab.lookup_indices([user_id])[0]
        return token_indices, user_index

    def _pad_sentence(self, tokens):
        return tokens[: self.optimal_sentence_length] + [self.PAD] * max(
            0, self.optimal_sentence_length - len(tokens)
        )

    def class_weight(self):
        from collections import Counter

        labels = [self[i][2].item() for i in range(len(self))]
        counts = Counter(labels)
        return [count for _, count in sorted(counts.items())]
