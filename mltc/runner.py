import random

import numpy as np
import torch
from transformers import BertTokenizer

from .model import BertForMultiLabelSequenceClassification
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


if __name__ == "__main__":
    args = {
        "max_seq_length": 512,
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "learning_rate": 3e-5,
        "num_train_epochs": 4.0,
        "warmup_proportion": 0.1,
        "local_rank": -1,
        "seed": 1234567890,
        "gradient_accumulation_steps": 1,
    }

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    logger.info("Initialization…")
    tokenizer = get_tokenizer()
    processor = MultiLabelTextProcessor("mltc/data", tokenizer, logger)
    model = get_model(len(processor.labels))

    logger.info("Training…")
    trainer = ModelTrainer(args, processor, model, logger)
    trainer.prepare_training_data()
    trainer.fit()

    logger.info("Evaluating…")
    evaluator = ModelEvaluator(args, processor, model, logger)
    evaluator.prepare_eval_data()
    results = evaluator.eval()
