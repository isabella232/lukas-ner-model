import numpy as np
from sklearn.metrics import roc_curve, auc

import torch
from torch.utils.data import DataLoader

from .metrics import accuracy, accuracy_thresh, fbeta

import pandas as pd
from tqdm import tqdm


class ModelEvaluator:
    def __init__(self, args, processor, model, logger):
        self.args = args
        self.processor = processor
        self.model = model
        self.logger = logger

        self.device = "cpu"
        self.eval_dataloader: DataLoader

    def prepare_eval_data(self, file_name):
        eval_examples = self.processor.get_examples(file_name, "eval")
        eval_features = self.processor.convert_examples_to_features(
            eval_examples, self.args["max_seq_length"]
        )
        self.eval_dataloader = self.processor.pack_features_in_dataloader(
            eval_features, self.args["local_rank"], self.args["eval_batch_size"], "eval"
        )

    def eval(self):
        all_logits = None
        all_labels = None

        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for input_ids, input_mask, segment_ids, label_ids in self.eval_dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, segment_ids, input_mask, label_ids)
                tmp_eval_loss, logits = outputs[:2]

            # logits = logits.detach().cpu().numpy()
            # label_ids = label_ids.to('cpu').numpy()
            # tmp_eval_accuracy = accuracy(logits, label_ids)
            tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate(
                    (all_logits, logits.detach().cpu().numpy()), axis=0
                )

            if all_labels is None:
                all_labels = label_ids.detach().cpu().numpy()
            else:
                all_labels = np.concatenate(
                    (all_labels, label_ids.detach().cpu().numpy()), axis=0
                )

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        # ROC-AUC calcualation
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(len(self.processor.labels)):
            fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            all_labels.ravel(), all_logits.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        result = {
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            # 'loss': tr_loss/nb_tr_steps,
            "roc_auc": roc_auc,
        }

        self.save_result(result)
        return result

    def save_result(self, result):
        output_eval_file = "mltc/data/eval_results.txt"
        with open(output_eval_file, "w") as writer:
            self.logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                self.logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    def predict(self, file_name):
        test_examples = self.processor.get_examples(file_name, "test")
        test_features = self.processor.convert_examples_to_features(
            test_examples, self.args["max_seq_length"]
        )
        test_dataloader = self.processor.pack_features_in_dataloader(
            test_features, self.args["local_rank"], self.args["eval_batch_size"], "test"
        )

        # Hold input data for returning it
        input_data = [
            {"id": input_example.guid, "text": input_example.text_a}
            for input_example in test_examples
        ]

        self.logger.info("***** Running prediction *****")
        self.logger.info("  Num examples = %d", len(test_examples))
        self.logger.info("  Batch size = %d", self.args["eval_batch_size"])

        all_logits = None
        self.model.eval()

        nb_eval_steps, nb_eval_examples = 0, 0
        for step, batch in enumerate(
            tqdm(test_dataloader, desc="Prediction Iteration")
        ):
            input_ids, input_mask, segment_ids = batch
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, segment_ids, input_mask)
                logits = outputs[0]
                logits = logits.sigmoid()

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate(
                    (all_logits, logits.detach().cpu().numpy()), axis=0
                )

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        label_names = [
            "Konst, kultur och nöje",
            "Brott, lag och rätt",
            "Katastrofer och olyckor",
            "Ekonomi, affärer och finans",
            "Utbildning",
            "Miljö och natur",
            "Medicin och hälsa",
            "Mänskligt",
            "Arbete",
            "Fritid och livsstil",
            "Politik",
            "Etik och religion",
            "Teknik och vetenskap",
            "Samhälle",
            "Sport",
            "Krig, konflikter och oroligheter",
            "Väder",
        ]

        return pd.merge(
            pd.DataFrame(input_data),
            pd.DataFrame(all_logits, columns=label_names),
            left_index=True,
            right_index=True,
        )
