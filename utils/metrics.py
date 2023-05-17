import torch
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


@torch.no_grad()
def max_pooling(
    embedding: torch.Tensor, embedding_layer: torch.nn.Module
) -> torch.Tensor:
    return embedding_layer(embedding).max(dim=0)[0]


@torch.no_grad()
def mean_pooling(
    embedding: torch.Tensor, embedding_layer: torch.nn.Module
) -> torch.Tensor:
    return embedding_layer(embedding).mean(dim=0)


# 获取每个item经过预加载模型的embedding表示
@torch.no_grad()
def get_item_embeddings(
    content,
    embedding_layer: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: str,
    pooling_func,
    data_type: str,
    only_id=False,
) -> torch.Tensor:
    # 填充一下为空的第0个item
    content[0] = {
        "title": "PAD",
        "genre": "PAD",
        "genres": "PAD",
        "category": "PAD",
        "desc": "PAD",
        "subcategory": "PAD",
        "abstract": "PAD",
        "description": "PAD",
    }

    item_embeddings = []
    for i, item_info in enumerate(content):
        if only_id:
            tokens = tokenizer(str(i))["input_ids"]
        else:
            if data_type == "mind":
                tokens = tokenizer(item_info["category"] + " : " + item_info["title"])[
                    "input_ids"
                ]
            elif data_type == "ml":
                tokens = (
                    tokenizer(item_info["genres"] + " : " + item_info["title"])["input_ids"]
                    if item_info
                    else tokenizer("PAD")["input_ids"]
                )
            elif data_type == "vg" or data_type == "mt" or data_type == "et":
                if item_info.get("title", False):
                    # if item_info.get("category", False):
                    #     tokens = tokenizer(item_info["category"] + " : " + item_info["title"])[
                    #         "input_ids"
                    #     ]
                    # else:
                    tokens = tokenizer(item_info["title"])["input_ids"]
                else:
                    tokens = tokenizer("PAD")["input_ids"]
            elif data_type in ["children", "comics_graphic", "poetry"]:
                tokens = (
                    tokenizer(item_info["genre"] + " : " + item_info["title"])["input_ids"]
                    if item_info
                    else tokenizer("PAD")["input_ids"]
                )
        # tokens = (
        #     tokenizer(item_info["title"])["input_ids"]
        #     if item_info
        #     else tokenizer("PAD")["input_ids"]
        # )
        embedding = torch.tensor(tokens).to(device)
        item_embeddings.append(pooling_func(embedding, embedding_layer))
    item_embeddings = torch.stack(item_embeddings, dim=0)

    return item_embeddings.to(device)


class ResultDataset(Dataset):
    def __init__(self, preds, labels) -> None:
        assert len(preds) == len(labels)
        super().__init__()
        self.preds = preds
        self.labels = labels

    def __len__(self):
        return len(self.preds)

    # 去重
    def unique(self, x):
        d = {}
        for i in range(len(x)):
            if d.get(x[i], False):
                x[i] = 0
            else:
                d[x[i]] = True
        return np.delete(x, np.where(x == 0))

    def __getitem__(self, index):
        pred = np.array(self.preds[index])
        label = np.array(self.labels[index])

        pred = self.unique(pred)
        if len(pred) < 10:
            pred = np.pad(pred, (0, 10 - len(pred)), mode="constant")

        # 计算预测是否正确
        y_true = np.where(pred == label, 1, 0)
        order = [i for i in range(10)]

        # 计算单个样本的指标
        hit1 = np.sum(y_true[0])
        hit5 = np.sum(y_true[0:5])
        hit10 = np.sum(y_true[0:10])
        ndcg5 = _ndcg_score(y_true, order, 5)
        ndcg10 = _ndcg_score(y_true, order, 10)
        return {
            "Hit@1": hit1,
            "Hit@5": hit5,
            "Hit@10": hit10,
            "nDCG@5": ndcg5,
            "nDCG@10": ndcg10,
        }


def _dcg_score(y_true, order, k=10):
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def _ndcg_score(y_true, order, k=10):
    actual = _dcg_score(y_true, order, k)
    return actual / 1.0


def metric(pred_ids, labels, log=True):
    ResDataset = ResultDataset(pred_ids, labels)
    ResDataLoader = DataLoader(ResDataset, batch_size=32, num_workers=0)

    hit1, hit5, hit10, total = 0, 0, 0, len(ResDataset)
    ndcg5, ndcg10 = [], []
    for res in ResDataLoader:
        hit1 += torch.sum(res["Hit@1"]).to(int)
        hit5 += torch.sum(res["Hit@5"]).to(int)
        hit10 += torch.sum(res["Hit@10"]).to(int)

        ndcg5.append(res["nDCG@5"])
        ndcg10.append(res["nDCG@10"])

    ndcg5 = torch.cat(ndcg5, dim=0)
    ndcg10 = torch.cat(ndcg10, dim=0)

    if log:
        logging.info(
            "Evaluation result:\nHit@1: {:.4f}\nHit@5: {:.4f}\nHit@10: {:.4f}\nnDCG@5: {:.4f}\nnDCG@10: {:.4f}\n".format(
                hit1 / total,
                hit5 / total,
                hit10 / total,
                torch.mean(ndcg5),
                torch.mean(ndcg10),
            )
        )

    print(
        "Hit@1: {:.4f}\nHit@5: {:.4f}\nHit@10: {:.4f}\nnDCG@5: {:.4f}\nnDCG@10: {:.4f}\n".format(
            hit1 / total,
            hit5 / total,
            hit10 / total,
            torch.mean(ndcg5),
            torch.mean(ndcg10),
        )
    )
    return torch.mean(ndcg10)