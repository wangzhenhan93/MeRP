# src/llamafactory/data/simple_collator.py
from typing import List, Dict, Any
import torch
from torch.nn.utils.rnn import pad_sequence
from ..extras.constants import IGNORE_INDEX

class SimpleCollator:
    def __init__(self, tokenizer, pad_token_id=None, ignore_index: int = IGNORE_INDEX):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id if pad_token_id is not None else getattr(tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(tokenizer, "eos_token_id", 0)
        self.ignore_index = ignore_index

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        for i, f in enumerate(features):
            if "input_ids" not in f:
                raise ValueError(f"Feature {i} missing 'input_ids'")
            input_ids_list.append(torch.tensor(f["input_ids"], dtype=torch.long))
            if "labels" in f and f["labels"] is not None:
                labels_list.append(torch.tensor(f["labels"], dtype=torch.long))
            else:
                labels_list.append(torch.tensor([], dtype=torch.long))

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        if labels_list and labels_list[0].numel() > 0:
            labels = pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index)
            if labels.size(1) < input_ids.size(1):
                pad_len = input_ids.size(1) - labels.size(1)
                labels = torch.cat([labels, torch.full((labels.size(0), pad_len), fill_value=self.ignore_index, dtype=torch.long)], dim=1)
        else:
            labels = torch.full_like(input_ids, fill_value=self.ignore_index)

        attention_mask = (input_ids != self.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
