import copy
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def collate_fn_train(batch):
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attn_masks"] for x in batch]
    id_seqs = [x["id_seq"] for x in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "id_seqs": torch.stack(id_seqs),
    }


def collate_fn_test(batch):
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attn_masks"] for x in batch]
    labels = [x["label"] for x in batch]
    id_seqs = [x["id_seq"] for x in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "label": labels,
        "id_seqs": torch.stack(id_seqs),
    }


class ReasoningDataset(Dataset):
    r"""
    A Dataset Class for building Dataloader of MIND or other datasets.
    """

    def __init__(
        self,
        behavior,
        content,
        tokenizer: AutoTokenizer,
        mode: str,
        data_type="",
        max_item=10,
        min_item=5,
        per_max_token=32,
        encoder_max_length=512,
        split_text=[],
        prefix_text="",
        autoregressive=True,
    ) -> None:
        assert mode in ["train", "dev", "test"]
        super().__init__()
        self.behavior = self.process_behavior(behavior, max_item, min_item, mode)
        self.content = content
        self.tokenizer = tokenizer
        self.mode = mode
        self.data_type = data_type
        self.max_item = max_item
        self.max_token = per_max_token  # max tokens of per item
        self.max_length = encoder_max_length
        self.autoregressive = autoregressive
        if not split_text:
            self.split_tokens = []
        elif type(split_text) == str:
            self.split_tokens = tokenizer(split_text)["input_ids"]
        elif type(split_text) == list:
            self.split_tokens = split_text
        else:
            self.split_tokens = []
        self.prefix_tokens = tokenizer(prefix_text)["input_ids"] if prefix_text else []

    def process_behavior(self, behavior, max_item, min_item, mode):
        r"""
        Truncate too long ( > max_item) and delete too short( < min_item) behaviors.
        """
        processed_behavior = []
        for i in behavior:
            user_id, history = i[0], i[1:]

            if len(history) < min_item:
                continue

            processed_behavior.append([user_id] + history)
        return processed_behavior

    def __len__(self):
        return len(self.behavior)

    def __getitem__(self, index):
        user_id, history = self.behavior[index][0], self.behavior[index][1:]
        if self.mode == "train" or self.mode == "dev":
            history = history[0:-1]

        label = history.pop()
        if len(history) > self.max_item:
            history = history[len(history) - self.max_item :]

        id_seq = (self.max_item - (len(history))) * [0] + history

        if self.autoregressive:
            if self.mode == "train" or self.mode == "dev":
                history.append(label)

            input_ids = self.template(
                user_id,
                history,
            )
        else:
            input_ids = self.template_noar(history, label)

        attn_masks = [1] * len(input_ids)
        input_ids, attn_masks = self.padding(input_ids, attn_masks)

        if self.mode == "train":
            return {
                "input_ids": torch.LongTensor(input_ids),
                "attn_masks": torch.FloatTensor(attn_masks),
                "id_seq": torch.LongTensor(id_seq),
            }
        # Testing needs to use the item ID of the predicted ground truth
        elif self.mode == "dev" or self.mode == "test":
            return {
                "input_ids": torch.LongTensor(input_ids),
                "attn_masks": torch.FloatTensor(attn_masks),
                "label": label,
                "id_seq": torch.LongTensor(id_seq),
            }
    
    def get_item_title(self, item_id: int, only_id=False):
        if only_id:
            item_title = str(item_id)
        else:
            #Different ways of handling different datasets
            if self.data_type == "mind":
                item_title = (
                    self.content[item_id]["category"]
                    + " : "
                    + self.content[item_id]["title"]
                )
            elif self.data_type == "ml":
                item_title = (
                    self.content[item_id]["genres"]
                    + " : "
                    + self.content[item_id]["title"]
                )
            elif self.data_type in ["vg", "mt", "et"]:
                item_title = self.content[item_id]["title"]
            elif self.data_type in ["children", "comics_graphic", "poetry"]:
                item_title = (
                    self.content[item_id]["genre"]
                    + " : "
                    + self.content[item_id]["title"]
                )
        return item_title

    def template(
        self,
        user_id: int,
        history,
    ):
        r"""
        The contents of the items are stitched together according to a template to construct the input.
        """
        input_ids = self.prefix_tokens.copy()
        split_token = self.split_tokens.copy()
        for item_id in history:
            item_title = self.get_item_title(item_id)
            # item_title = self.content[item_id]["title"]
            tokens = self.tokenizer(item_title)["input_ids"]

            # Trunate more than the maximum length of a single
            if len(tokens) > self.max_token:
                tokens = tokens[0 : self.max_token]

            input_ids.extend(tokens + split_token)
        return (
            input_ids[0 : -len(split_token)]
            if self.mode == "train" and split_token
            else input_ids
        )
    
    def template_noar(self, history: list, label: int):
        input_ids = self.prefix_tokens.copy()
        split_token = self.split_tokens.copy()
        for item_id in history:
            item_title = self.get_item_title(item_id)
            # item_title = self.content[item_id]["title"]
            tokens = self.tokenizer(item_title)["input_ids"]

            # Trunate more than the maximum length of a single
            if len(tokens) > self.max_token:
                tokens = tokens[0 : self.max_token]

            input_ids.extend(tokens + split_token)

        if split_token:
            input_ids = input_ids[0 : -len(split_token)]

        reason_tokens = self.tokenizer("Now the user may want to watch ")["input_ids"]
        input_ids.extend(reason_tokens)

        if self.mode == "train":
            label_tokens = self.tokenizer(self.get_item_title(label))["input_ids"]
            input_ids.extend(label_tokens)

        return input_ids


    def padding(self, input_ids: list, attn_masks: list):
        r"""
        Padding the inputs for GPT model.

        For training, we pad the right side,
        For testing, we pad the left side.
        """
        assert len(input_ids) <= self.max_length

        if self.mode == "train":
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (
                self.max_length - len(input_ids)
            )
            attn_masks = attn_masks + [0] * (self.max_length - len(attn_masks))
        elif self.mode == "dev" or self.mode == "test":
            input_ids = [self.tokenizer.pad_token_id] * (
                self.max_length - len(input_ids)
            ) + input_ids
            attn_masks = [0] * (self.max_length - len(attn_masks)) + attn_masks
        return input_ids, attn_masks
