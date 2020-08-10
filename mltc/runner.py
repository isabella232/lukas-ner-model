import random

import numpy as np
import torch
from transformers import BertTokenizer

from .models import (
    BertForMultiLabelSequenceClassification,
    BertForMultilabelSpecialized,
)
from .processor import MultiLabelTextProcessor
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Prepare model
def get_base_model(num_labels):
    BASE_PATH = "mltc/data/model_files/"
    config = BASE_PATH + "config.json"
    model_state_dict = torch.load(BASE_PATH + "finetuned_2020-08-10_pytorch_model.bin")

    model = BertForMultiLabelSequenceClassification.from_pretrained(
        config, num_labels=num_labels, state_dict=model_state_dict, num_labels_parent=17
    )

    return model


def get_specialized_model(num_labels):
    BASE_PATH = "mltc/data/model_files/"
    config = BASE_PATH + "config.json"
    model_state_dict = torch.load(BASE_PATH + "finetuned_2020-08-03_pytorch_model.bin")
    del model_state_dict["classifier.weight"]
    del model_state_dict["classifier.bias"]

    model = BertForMultilabelSpecialized.from_pretrained(
        config, num_labels=num_labels, state_dict=model_state_dict
    )

    print(model.modules)
    [print(name, param.requires_grad) for name, param in model.named_parameters()]

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


if __name__ == "__main__":
    args = {
        "max_seq_length": 512,
        "num_train_epochs": 4.0,
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "learning_rate": 3e-5,
        "warmup_proportion": 0.1,
        "seed": 1234567890,
        "do_train": False,
    }

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    logger.info("Initializing…")
    tokenizer = get_tokenizer()
    processor = MultiLabelTextProcessor(tokenizer, logger, "top_categories.txt")
    model = get_base_model(len(processor.labels))

    if args["do_train"]:
        logger.info("Training…")
        trainer = ModelTrainer(args, processor, model, logger)
        trainer.prepare_training_data(
            "train_small.csv", "train_small_parent_labels.csv"
        )
        trainer.train()

    logger.info("Evaluating…")
    evaluator = ModelEvaluator(args, processor, model, logger)
    evaluator.prepare_eval_data("eval_small.csv", "eval_small_parent_labels.csv")
    results = evaluator.evaluate()
    print(results)

    # results = evaluator.predict("test.csv")
    # TODO: kika layer size
    # TODO: jämföra epoch eval loss för att hitta optimalt antal träningsepoker
    # TODO: sätta alla för små kategori i en övrigt-kategori

