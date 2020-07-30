import random

import numpy as np

from .model import BertForMultiLabelSequenceClassification
from .processor import MultiLabelTextProcessor
from .trainer import ModelTrainer


from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup


import torch
from torch import Tensor
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Prepare model
def get_model(num_labels):
    BASE_PATH = "mltc/data/model_files/"
    config = BASE_PATH + "config.json"
    model_state_dict = torch.load(BASE_PATH + "pytorch_model.bin")

    model = BertForMultiLabelSequenceClassification.from_pretrained(
        config, num_labels=num_labels, state_dict=model_state_dict,
    )

    return model


def get_tokenizer():
    tokenizer = BertTokenizer(
        vocab_file="mltc/data/model_files/vocab.txt",
        do_lower_case=False,
        strip_accents=False,
        keep_accents=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )

    return tokenizer


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def prepare_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args["learning_rate"], correct_bias=False
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args["warmup_proportion"] * args["num_train_steps"],
        num_training_steps=args["num_train_steps"],
    )

    # scheduler = CyclicLR(
    #     optimizer,
    #     base_lr=2e-5,
    #     max_lr=5e-5,
    #     step_size_up=2500,
    #     last_epoch=0,
    #     cycle_momentum=False,
    # )

    return optimizer, scheduler


def fit(args, model, train_dataloader):

    optimizer, scheduler = prepare_optimizer(args, model)

    device = "cpu"

    global_step = 0
    model.train()
    for i_ in tqdm(range(int(args["num_train_epochs"])), desc="Epoch"):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            outputs = model(input_ids, segment_ids, input_mask, label_ids)
            loss = outputs[0]

            if args["gradient_accumulation_steps"] > 1:
                loss = loss / args["gradient_accumulation_steps"]

            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args["gradient_accumulation_steps"] == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args["learning_rate"] * warmup_linear(
                    global_step / args["num_train_steps"], args["warmup_proportion"]
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        logger.info("Loss after epoc {}".format(tr_loss / nb_tr_steps))
        logger.info("Eval after epoc {}".format(i_ + 1))

    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Only save the model it-self
    output_model_file = "mltc/data/model_files/finetuned_pytorch_model.bin"
    torch.save(model_to_save.state_dict(), output_model_file)
    # eval()


if __name__ == "__main__":
    args = {
        "max_seq_length": 512,
        "do_train": True,
        "do_eval": True,
        "do_lower_case": True,
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "learning_rate": 3e-5,
        "num_train_epochs": 4.0,
        "warmup_proportion": 0.1,
        "no_cuda": False,
        "local_rank": -1,
        "seed": 1234567890,
        "gradient_accumulation_steps": 1,
        "optimize_on_cpu": False,
        "loss_scale": 128,
    }

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    trainer = ModelTrainer(args, tokenizer)
    trainer.prepare_training_data()
    trainer.fit(args, model)

    label_list = trainer.processor.labels
    num_labels = len(label_list)

    tokenizer = get_tokenizer()
    model = get_model(num_labels)
    device = "cpu"
    model.to(device)

    logger.info("Fitting…")
    fit(args, model, train_dataloader)

