import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm


class ModelTrainer:
    def __init__(self, args, processor, model, logger):
        self.args = args
        self.processor = processor
        self.model = model
        self.logger = logger

        self.device = "cpu"
        self.train_dataloader: DataLoader
        self.num_train_steps: int

        self.optimizer: AdamW
        self.scheduler: LambdaLR

    def prepare_training_data(self):
        train_examples = self.processor.get_train_examples()

        self.num_train_steps = int(
            len(train_examples)
            / self.args["train_batch_size"]
            / self.args["gradient_accumulation_steps"]
            * self.args["num_train_epochs"]
        )

        train_features = self.processor.convert_examples_to_features(
            train_examples, self.args["max_seq_length"]
        )
        self.train_dataloader = self.processor.pack_features_in_dataloader(
            train_features, self.args["local_rank"], self.args["train_batch_size"], True
        )

    def warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def prepare_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
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

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args["learning_rate"],
            correct_bias=False,
        )

    def prepare_scheduler(self):
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args["warmup_proportion"] * self.num_train_steps,
            num_training_steps=self.num_train_steps,
        )

        # scheduler = CyclicLR(
        #     optimizer,
        #     base_lr=2e-5,
        #     max_lr=5e-5,
        #     step_size_up=2500,
        #     last_epoch=0,
        #     cycle_momentum=False,
        # )

    def fit(self):
        self.model.to(self.device)
        self.prepare_optimizer()
        self.prepare_scheduler()

        global_step = 0
        self.model.train()
        for i_ in tqdm(range(int(self.args["num_train_epochs"])), desc="Epoch"):

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                outputs = self.model(input_ids, segment_ids, input_mask, label_ids)
                loss = outputs[0]

                if self.args["gradient_accumulation_steps"] > 1:
                    loss = loss / self.args["gradient_accumulation_steps"]

                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.args["gradient_accumulation_steps"] == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = self.args["learning_rate"] * self.warmup_linear(
                        global_step / self.num_train_steps,
                        self.args["warmup_proportion"],
                    )
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr_this_step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    global_step += 1

            self.logger.info(f"Loss after epoch {i_+1}: {tr_loss / nb_tr_steps}")
            self.logger.info(
                f"Learning rate after epoch {i_+1}: {self.scheduler.get_last_lr()[0]}"
            )
            # self.logger.info(f"Eval after epoc {i_ + 1}")

        self.model.save()

    def save_model(self):
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model it-self
        output_model_file = "mltc/data/model_files/finetuned_pytorch_model.bin"
        torch.save(model_to_save.state_dict(), output_model_file)
