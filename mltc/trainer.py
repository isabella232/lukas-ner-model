from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .processor import MultiLabelTextProcessor


class ModelTrainer:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.processor = MultiLabelTextProcessor("mltc/data", tokenizer)
        self.train_dataloader: DataLoader

    def prepare_training_data(self):
        train_examples = self.processor.get_train_examples()
        # eval_examples = processor.get_dev_examples()

        num_train_steps = int(
            len(train_examples)
            / self.args["train_batch_size"]
            / self.args["gradient_accumulation_steps"]
            * self.args["num_train_epochs"]
        )

        train_features = self.processor.convert_examples_to_features(
            train_examples, self.args["max_seq_length"]
        )
        self.train_dataloader = self.processor.pack_features_in_dataloader(
            train_features, self.args["local_rank"], self.args["train_batch_size"]
        )

