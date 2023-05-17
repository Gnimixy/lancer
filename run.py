import os
import sys
import copy
import json
import math
import time
import torch
import pickle
import random
import logging
import transformers
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from layer.prefix import PrefixPrompt
from layer.attention import ReasoningAttention
from utils.args import get_hyperparams
from utils.k_dataset import KnowledgeDataset, collate_fn_k
from utils.r_dataset import ReasoningDataset, collate_fn_train, collate_fn_test
from utils.metrics import mean_pooling, max_pooling, get_item_embeddings, metric


# Set random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


# load ndjson
def load_ndjson(input_file):
    with open(input_file, "r") as f:
        lines = f.read()
        d = [json.loads(l) for l in lines.splitlines()]
    return d


# load sequential_data.txt
def load_seq_txt(input_file):
    output = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            line = line.split(" ")
            line = [int(i) for i in line]
            output.append(line)
    return output


@autocast()
def knowledge(item_ids, k_dataset, prompt, model_k, seq_item_hss, labels=True):
    """
    Knowledge prompt for learning the domain knowledge of the recommendation system.

    Input user/item interaction history list.
    Output CE loss (if trainning) and hidden states of items.

    Args:
        k_dataset (`KnowledgeDataset`):
            Dataset of detailed information on the contents of the item.
        prompt (`PrefixPrompt`):
            Implementation of prefix prompt.
        model_k (`GPT2LMHeadModel`):
            Autoregressive language model of knowledge.
        seq_item_hss (`list`):
            Hidden states of sequence of items.
    """
    item_info = []
    seq_item_hs = []
    for item_id in item_ids:
        if item_id == 0:
            seq_item_hs.append(
                torch.zeros(
                    (1, model_r.config.n_embd),
                    device=args.device,
                )
            )
        else:
            item_info.append(k_dataset[item_id])
    item_info = collate_fn_k(item_info)

    past_key_values, attention_mask_ = prompt(
        item_info["attention_mask"].to(args.device), output="tuple"
    )

    outputs = model_k(
        input_ids=item_info["input_ids"].to(args.device),
        attention_mask=attention_mask_,
        output_hidden_states=True,
        labels=item_info["input_ids"].to(args.device) if labels else None,
        past_key_values=past_key_values,
    )

    last_hidden_state = outputs.hidden_states[0]
    # num_item, 512, 768
    last_hidden_state = last_hidden_state[:, -1, :]  # num_item, 1, 768
    seq_item_hs.append(last_hidden_state.detach())

    seq_item_hs = torch.cat(seq_item_hs, dim=0)
    seq_item_hss.append(seq_item_hs)
    return outputs.loss, seq_item_hss


@autocast()
def reasoning(seq_item_hss, batch_size, attention_mask, num_beams=1):
    seq_item_hss = torch.stack(seq_item_hss).view(batch_size, args.max_item, -1)

    past_key_values, _ = prompt(attention_mask, output="hidden_state")

    past_key_values = attn_layer(seq_item_hss, past_key_values, past_key_values.clone())

    past_key_values = past_key_values.view(
        batch_size,
        args.max_item,
        2,
        model_r.config.n_layer,
        model_r.config.n_head,
        model_r.config.n_embd // model_r.config.n_head,
    )
    if num_beams > 1:
        past_key_values = past_key_values.repeat_interleave(num_beams, dim=0)

    past_key_values = past_key_values.permute(3, 2, 0, 4, 1, 5)
    past_key_values = tuple(tuple(j for j in i) for i in past_key_values)

    prefix_attention_mask = torch.ones(batch_size, args.max_item, device=args.device)
    attention_mask_ = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    if num_beams == 1:
        outputs = model_r(
            input_ids=input_ids,
            attention_mask=attention_mask_,
            labels=input_ids,
            past_key_values=past_key_values,
        )
        return outputs
    else:
        return past_key_values, attention_mask_


# model test and evaluate
def test(model_k, model_r, prompt, attn_layer, TestDataLoader, args):
    model_r.eval()
    model_k.eval()
    attn_layer.eval()
    prompt.eval()
    with torch.no_grad():
        # Get the embeddings corresponding to each item
        embedding_layer = model_r.get_input_embeddings()
        item_embeddings = get_item_embeddings(
            content,
            embedding_layer,
            tokenizer,
            args.device,
            max_pooling,
            args.data_type,
        )

        pred_ids = []
        labels = []
        for step, data in enumerate(tqdm(TestDataLoader, desc="Eval", ncols=120)):
            input_ids = data["input_ids"].to(args.device)
            attention_mask = data["attention_mask"].to(args.device)
            label = data["label"]  # type: list
            id_seqs = data["id_seqs"]
            batch_size = input_ids.size()[0]

            with autocast():
                seq_item_hss = []
                for item_ids in id_seqs:
                    _, seq_item_hss = knowledge(
                        item_ids, k_dataset, prompt, model_k, seq_item_hss, labels=False
                    )

                past_key_values, attention_mask_ = reasoning(
                    seq_item_hss, batch_size, attention_mask, num_beams=args.num_beams
                )

                # model generate
                outputs = model_r.generate(
                    input_ids,
                    attention_mask=attention_mask_,
                    prefix_key_value=past_key_values,
                    pad_token_id=tokenizer.pad_token_id,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_return_sequences,
                    do_sample=False,
                    max_new_tokens=args.per_max_token,
                )

                # Take the generated part and discard the prompt
                outputs = outputs[:, args.encoder_max_length :]

                cos_sims = []
                for output in outputs:
                    # Take the part before [eos]
                    output = output[output != tokenizer.eos_token_id]

                    # Split the sentence before the delimiter (select the first one if multiple items are generated)
                    if split_token and split_token in output:
                        output = output[: torch.where(output == split_token)[0][0]]

                    # The embedding of the generated item
                    embedding = max_pooling(output, embedding_layer)

                    # Calculate similarity
                    cos_sims.append(
                        torch.cosine_similarity(embedding, item_embeddings, dim=-1)
                    )
                cos_sims = torch.stack(cos_sims)
                # [batch_size * NUM_RETURN_SEQUENCES, num_items]

                # Select the item with the greatest similarity
                pred_id = torch.argmax(cos_sims, dim=-1).reshape(
                    -1, args.num_return_sequences
                )

            pred_ids.extend(pred_id.detach().cpu().tolist())
            labels.extend(label)

            # if (step + 1) % 2000 == 0:
            #     logging.info(f"Step: {step+1}")
            #     break

        res = metric(pred_ids, labels)
    return res


if __name__ == "__main__":
    args = get_hyperparams()

    seed_everything(args.seed)

    # Initialize the model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model_r = GPT2LMHeadModel.from_pretrained("gpt2").to(args.device)
    model_k = GPT2LMHeadModel.from_pretrained("gpt2").to(args.device)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    # Initialize knowledge prompt
    prompt = PrefixPrompt(
        n_layer=model_r.config.n_layer,
        n_embd=model_r.config.n_embd,
        n_head=model_r.config.n_head,
        seq_len=args.prefix_length,
        device=args.device,
    )
    prompt = prompt.to(args.device)

    # Load dataset
    if args.data_type == "ml":
        DATA_DIR = "./data/ml-1m"
    elif args.data_type == "mind":
        DATA_DIR = "./data/MINDsmall"
    elif args.data_type == "poetry":
        DATA_DIR = "./data/Goodreads/poetry"
    behavior = load_seq_txt(os.path.join(DATA_DIR, "sequential_data.txt"))
    content = load_ndjson(os.path.join(DATA_DIR, "content.json"))

    if args.data_type == "ml":
        PREFIX_TEXT = "A user watched sequence of movies : "
    elif args.data_type == "mind":
        PREFIX_TEXT = "News reading sequence: "
    elif args.data_type == "poetry":
        PREFIX_TEXT = "A user liked the following books: "
    split_tokens = tokenizer(args.split_text)["input_ids"]
    split_token = split_tokens[0]

    attn_layer = ReasoningAttention().to(args.device)

    logging.basicConfig(
        filename=os.path.join(args.model_path, "log.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    TrainDataset = ReasoningDataset(
        behavior,
        content,
        tokenizer,
        mode="train",
        data_type=args.data_type,
        max_item=args.max_item,
        min_item=args.min_item,
        per_max_token=args.per_max_token,
        encoder_max_length=args.encoder_max_length,
        split_text=args.split_text,
        prefix_text=PREFIX_TEXT,
    )
    TrainDataLoader = DataLoader(
        TrainDataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_train,
    )
    TestDataset = ReasoningDataset(
        behavior,
        content,
        tokenizer,
        mode="test",
        data_type=args.data_type,
        max_item=args.max_item,
        min_item=args.min_item,
        per_max_token=args.per_max_token,
        encoder_max_length=args.encoder_max_length,
        split_text=args.split_text,
        prefix_text=PREFIX_TEXT,
    )
    TestDataLoader = DataLoader(
        TestDataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_test,
    )
    k_dataset = KnowledgeDataset(
        content,
        tokenizer,
        encoder_max_length=args.encoder_max_length,
        data_type=args.data_type,
        task="random",
    )

    for param in model_r.parameters():
        param.requires_grad = False
    for param in model_k.parameters():
        param.requires_grad = False
    for param in prompt.parameters():
        param.requires_grad = True
    for param in attn_layer.parameters():
        param.requires_grad = True

    param_dict = [
        {"params": prompt.parameters(), "lr": args.lr},
        {"params": attn_layer.parameters(), "lr": args.lr},
    ]

    optimizer = torch.optim.Adam(param_dict)
    scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=args.f_epochs * len(TrainDataLoader),
    )
    scaler = GradScaler()

    # Model training
    best = 0.0
    for epoch in range(args.f_epochs):
        step, train_losses = 0, 0.0
        tqdm_iter = tqdm(TrainDataLoader, desc=f"K Epoch {epoch+1}", ncols=120)
        model_r.train()
        model_k.eval()
        attn_layer.train()
        prompt.train()
        for data in tqdm_iter:
            input_ids = data["input_ids"].to(args.device)
            attention_mask = data["attention_mask"].to(args.device)
            id_seqs = data["id_seqs"].to(args.device)
            batch_size = input_ids.size()[0]

            with autocast():
                seq_item_hss = []
                for item_ids in id_seqs:
                    loss, seq_item_hss = knowledge(
                        item_ids, k_dataset, prompt, model_k, seq_item_hss, labels=True
                    )

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                outputs = reasoning(seq_item_hss, batch_size, attention_mask)

            scaler.scale(outputs.loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            optimizer.zero_grad()

            loss_value = outputs.loss.cpu().item()
            train_losses += loss_value
            step += 1
            tqdm_iter.set_postfix({"loss": format(train_losses / step, ".4f")})

        final_loss = format(train_losses / step, ".4f")
        logging.info(f"Epoch {epoch+1}\nLoss: {final_loss}")

    for param in model_r.parameters():
        param.requires_grad = True
    for param in model_k.parameters():
        param.requires_grad = False
    for param in prompt.parameters():
        param.requires_grad = True
    for param in attn_layer.parameters():
        param.requires_grad = True

    param_dict = [
        {"params": prompt.parameters(), "lr": args.lr},
        {"params": model_r.parameters(), "lr": args.lr, "betas": [0.9, 0.99]},
        {"params": attn_layer.parameters(), "lr": args.lr},
    ]

    optimizer = torch.optim.Adam(param_dict)
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=args.p_epochs * len(TrainDataLoader),
    )
    scaler = GradScaler()

    # Model training
    best = 0.0
    for epoch in range(args.p_epochs):
        step, train_losses = 0, 0.0
        tqdm_iter = tqdm(TrainDataLoader, desc=f"R Epoch {epoch+1}", ncols=120)
        model_r.train()
        model_k.eval()
        attn_layer.train()
        prompt.train()
        for data in tqdm_iter:
            input_ids = data["input_ids"].to(args.device)
            attention_mask = data["attention_mask"].to(args.device)
            id_seqs = data["id_seqs"].to(args.device)
            batch_size = input_ids.size()[0]

            with autocast():
                seq_item_hss = []
                for item_ids in id_seqs:
                    _, seq_item_hss = knowledge(
                        item_ids, k_dataset, prompt, model_k, seq_item_hss, labels=False
                    )

                outputs = reasoning(seq_item_hss, batch_size, attention_mask)

            scaler.scale(outputs.loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            loss_value = outputs.loss.cpu().item()
            train_losses += loss_value
            step += 1
            tqdm_iter.set_postfix({"loss": format(train_losses / step, ".4f")})

        final_loss = format(train_losses / step, ".4f")
        logging.info(f"Epoch {epoch+1}\nLoss: {final_loss}")

        res = test(model_k, model_r, prompt, attn_layer, TestDataLoader, args)
        if res > best:
            MODEL_STORED_PATH = args.model_path + "/" + str(epoch + 1)
            best = res
            model_r.save_pretrained(MODEL_STORED_PATH)
            torch.save(
                prompt.state_dict(),
                os.path.join(MODEL_STORED_PATH, "prompt.pt"),
            )
            torch.save(
                attn_layer.state_dict(),
                os.path.join(MODEL_STORED_PATH, "attn.pt"),
            )
