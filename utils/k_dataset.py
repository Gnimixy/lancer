import torch
import random
import numpy as np
from torch.utils.data import Dataset


class KnowledgeDataset(Dataset):
    r"""
    A Dataset Class for training the knowledge prompt
    """

    def __init__(
        self, content, tokenizer, encoder_max_length=512, data_type="", task="",
    ) -> None:
        super().__init__()
        assert data_type in ["ml", "mind", "vg", "mt", "et", "children", "comics_graphic", "poetry"]
        self.tokenizer = tokenizer
        self.max_length = encoder_max_length
        self.data_type = data_type
        if self.data_type == "ml":
            self.metadata_name = "desc"
        elif self.data_type == "mind":
            self.metadata_name = "abstract"
        elif self.data_type == ["vg", "mt", "et", "children", "comics_graphic", "poetry"]:
            self.metadata_name = "description"
        self.task = task
        if task == "random" or task == "simple":
            self.content = content
        else:
            self.content = self.clean_content(content)

    def clean_content(self, content: list[dict]) -> list:
        r"""
        For some content without long element data, delete them from dataset.
        """
        content_new = []
        for i in content:
            if i is not None and i.get(self.metadata_name, False):
                intro = i.get(self.metadata_name, float("nan"))
                if type(intro) == float and np.isnan(intro):
                    pass
                else:
                    content_new.append(i)
        return content_new

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        input_ids, loss_begin, loss_end = self.template(self.content[index])
        attn_masks = [1] * len(input_ids)
        input_ids, attn_masks = self.padding(input_ids, attn_masks)
        loss_ids = [0] * len(attn_masks)
        for i in range(loss_begin, loss_end):
            loss_ids[i] = 1
        return {
            "input_ids": torch.LongTensor(input_ids),
            "attn_masks": torch.FloatTensor(attn_masks),
            "loss_ids": torch.LongTensor(loss_ids),
        }

    def template(self, content: dict) -> list[int]:
        if self.data_type == "ml":
            main_element = "title"
            long_element = "desc"
            other_element = ["genres"]
        elif self.data_type == "mind":
            main_element = "title"
            long_element = "abstract"
            other_element = ["category", "subcategory"]
        elif self.data_type in ["vg", "mt", "et"]:
            main_element = "title"
            long_element = "description"
            other_element = ["category", "brand", "price", "feature"]
        elif self.data_type in ["children", "comics_graphic", "poetry"]:
            main_element = "title"
            long_element = "description"
            other_element = ["genre", "authors", "average_rating", "publication_year"]
        else:
            raise Exception("Illegal data type.")
        
        if self.task == "simple":
            text1 = content[main_element]
            input_ids = self.tokenizer(text1)["input_ids"]
            if len(input_ids) > self.max_length:
                input_ids = input_ids[0 : self.max_length]
            loss_begin, loss_end = 0, len(input_ids) - 1
            return input_ids, loss_begin, loss_end
        elif self.task == "expand":
            # for expand, random the other_element, predict the long_element
            text1 = main_element + ": " + content[main_element]
            random.shuffle(other_element)
            for i in other_element:
                text1 += " " + i + ": " + content[i]
            text1 += long_element + ": " 
            text2 = content[long_element]
        elif self.task == "summary":
            # for summary, predict a random main_element or other_element
            text1 = long_element + ": " + content[long_element]
            other_element.append(main_element)
            pred_element = random.choice(other_element)
            other_element.remove(pred_element)
            random.shuffle(other_element)
            for i in other_element:
                text1 += " " + i + ": " + content[i]
            text1 += pred_element + ": " 
            text2 = content[pred_element]
        elif self.task == "random":
            # other_element.append(main_element)
            # pred_element = random.choice(other_element)
            # other_element.remove(pred_element)
            # random.shuffle(other_element)
            if self.data_type == "mind" or self.data_type == "ml":
                pred_element = main_element
                if type(content[long_element]) != float:
                    text1 = long_element + " : " + content[long_element]
                else:
                    text1 = ""
                for i in other_element:
                    text1 += " " + i + " : " + content[i]
                text1 += pred_element + " : " 
                text2 = content[pred_element]
            else:
                pred_element = main_element
                if content.get(long_element, False) and type(content[long_element]) != float:
                    text1 = long_element + " : " + content[long_element]
                    while len(text1) > 1000:
                        text1 = text1.split(".")
                        text1 = ".".join(text1[0:-1])
                else:
                    text1 = ""
                for i in other_element:
                    if content.get(i, False):
                        text1 += " " + i + " : " + content[i]
                text1 += pred_element + " : " 
                text2 = content[pred_element]

        input_ids = self.tokenizer(text1)["input_ids"]
        input_ids2 = self.tokenizer(text2)["input_ids"]

        if len(input_ids) > self.max_length - 64:
            input_ids = input_ids[0 : self.max_length - 64]

        loss_begin = len(input_ids)
        input_ids.extend(input_ids2)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[0 : self.max_length]

        loss_end = len(input_ids)
        return input_ids, loss_begin, loss_end

    def padding(self, input_ids: list, attn_masks: list):
        r"""
        Padding the inputs for GPT model.

        For training, we pad the right side,
        """
        assert len(input_ids) <= self.max_length
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (
            self.max_length - len(input_ids)
        )
        attn_masks = attn_masks + [0] * (self.max_length - len(attn_masks))
        return input_ids, attn_masks
    

def collate_fn_k(batch):
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attn_masks"] for x in batch]
    loss_ids = [x["loss_ids"] for x in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "loss_ids": torch.stack(loss_ids)
    }
