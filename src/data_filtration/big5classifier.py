import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoConfig,
    PreTrainedModel,
    EvalPrediction,
    set_seed,
)
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm
import wandb


class Big5PersonalityClassifier:
    def __init__(self, model_name, cache_dir, output_dir, seed=42):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.seed = seed
        set_seed(self.seed)

        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ["WANDB_API_KEY"] = ""
        os.environ["WANDB_PROJECT"] = ""
        os.environ["WANDB_WATCH"] = "false"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.labels = ['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']
        self.id2label = {idx:label for idx, label in enumerate(self.labels)}
        self.label2id = {label:idx for idx, label in enumerate(self.labels)}
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=5,
            cache_dir=cache_dir,
            problem_type="multi_label_classification",
            id2label=self.id2label,
            label2id=self.label2id
        )

    def prepare_data(self):
        dataset = load_dataset("Fatima0923/Automated-Personality-Prediction")

        # Demo Run ------
        # indices = range(0, 100)
        # dataset_dict = {
        #     "train": dataset["train"].select(indices),
        #     "validation": dataset["validation"].select(indices),
        #     "test": dataset["test"].select(indices)
        # }
        # dataset = DatasetDict(dataset_dict)
        # ------

        encoded_dataset = dataset.map(self.tokenize_and_encode, batched=True)
        encoded_dataset = encoded_dataset.remove_columns(
            ["conscientiousness", "agreeableness", "extraversion", "neuroticism", "openness", "text"]
        )
        encoded_dataset.set_format("torch")
        return encoded_dataset

    def tokenize_and_encode(self, batch):
        text = batch["text"]
        encoding = self.tokenizer(text, padding="max_length", truncation=True)
        traits = ['agreeableness', 'openness', 'conscientiousness', 'extraversion', 'neuroticism']
        for trait in traits:
            batch[trait] = [1 if x > 40 else 0 for x in batch[trait]]
        encoding['labels'] = [list(map(float, x)) for x in zip(*[batch[trait] for trait in traits])]
        return encoding

    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def multi_label_metrics(self, predictions, labels, threshold=0.4):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                'roc_auc': roc_auc,
                'accuracy': accuracy}
        return metrics

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions,
                tuple) else p.predictions
        result = self.multi_label_metrics(
            predictions=preds,
            labels=p.label_ids)
        return result

    def train(self):
        encoded_dataset = self.prepare_data()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy = "epoch",
            save_strategy = "epoch",
            save_total_limit=2,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            log_level="info",
            logging_steps=1,
            logging_strategy="steps",
            report_to="wandb",
            seed=self.seed
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=encoded_dataset['train'],
            eval_dataset=encoded_dataset['validation'],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        trainer.train()
        # self.model.save_pretrained(self.output_dir)
        # self.tokenizer.save_pretrained(self.output_dir)

    def predict(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        model = self.model.to(self.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.sigmoid(logits)
        return probs
    
    def predict_single(self, text):
        # Tokenize the input text
        encoding = self.tokenizer(text, return_tensors="pt")
        encoding = {k: v.to(self.device) for k,v in encoding.items()}

        # Get model outputs
        self.model.to(self.device)
        outputs = self.model(**encoding)
        logits = outputs.logits

        # Apply sigmoid and threshold
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.4)] = 1

        # Turn predicted id's into actual label names
        predicted_labels = [self.id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
        return predicted_labels

    def evaluate_test(self):
        encoded_dataset = self.prepare_data()
        test_dataset = encoded_dataset['test']
        batch_size = 64
        # data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
        # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

        all_labels = []
        all_preds = []

        acc, f1 = [], []

        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for batch in tqdm(test_dataset, desc="Evaluating", unit="batch"):
                input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
                labels = batch['labels'].unsqueeze(0).cpu().numpy()

                # Check input shapes before passing to model
                # print(f"input_ids shape: {input_ids.shape}")
                # print(f"attention_mask shape: {attention_mask.shape}")

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                sigmoid = torch.nn.Sigmoid()
                probs = sigmoid(logits.squeeze().cpu())
                predictions = np.zeros(probs.shape)
                predictions = (probs >= 0.4).float().numpy()

                all_labels.append(labels)
                all_preds.append(predictions)

        all_labels = np.vstack(all_labels) # n x 5
        all_preds = np.vstack(all_preds) # n x 5 

        for true_labels, predicted_labels in zip(all_labels, all_preds):
            acc.append(accuracy_score(true_labels, predicted_labels))
            f1.append(f1_score(true_labels, predicted_labels, average='micro'))

        overall_accuracy = sum(acc) / len(acc)
        overall_f1_score = sum(f1) / len(f1)

        pred_agreeableness, true_agreeableness = all_preds[:, 0], all_labels[:, 0]
        pred_openness, true_openness = all_preds[:, 1], all_labels[:, 1]
        pred_conscientiousness, true_conscientiousness = all_preds[:, 2], all_labels[:, 2]
        pred_extraversion, true_extraversion = all_preds[:, 3], all_labels[:, 3]
        pred_neuroticism, true_neuroticism = all_preds[:, 4], all_labels[:, 4]

        metrics = self.multi_label_metrics(all_preds, all_labels)

        # Save results to a file
        with open("./results.txt", "w") as f:
            f.write("----------------RESULTS---------------------------\n")
            f.write(f"Metric Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Metric F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"Metric ROC AUC: {metrics['roc_auc']:.4f}\n")
            f.write("\n")
            f.write(f"A Acc: {accuracy_score(true_agreeableness, pred_agreeableness)}, A F1: {f1_score(true_agreeableness, pred_agreeableness)}\n")
            f.write(f"O Acc: {accuracy_score(true_openness, pred_openness)}, O F1: {f1_score(true_openness, pred_openness)}\n")
            f.write(f"C Acc: {accuracy_score(true_conscientiousness, pred_conscientiousness)}, C F1: {f1_score(true_conscientiousness, pred_conscientiousness)}\n")
            f.write(f"E Acc: {accuracy_score(true_extraversion, pred_extraversion)}, E F1: {f1_score(true_extraversion, pred_extraversion)}\n")
            f.write(f"N Acc: {accuracy_score(true_neuroticism, pred_neuroticism)}, N F1: {f1_score(true_neuroticism, pred_neuroticism)}\n")
            f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
            f.write(f"Overall F1 Score: {overall_f1_score:.4f}\n")
        
        print(f"Results have been saved")


    def run(self, csv_file=None, evaluate_flag=False, text=None):
        if evaluate_flag and text:
            return self.predict_single(text)
        elif evaluate_flag:
            self.evaluate_test()
        else:
            self.train()

if __name__ == "__main__":

    # TRAIN ARGS ----
    # TRAIN_FLAG = True
    # model_name = 'distilbert/distilbert-base-uncased'
    # output_dir=f""
    # ----

    # EVAL ARGS ----
    TRAIN_FLAG = False
    model_name = ""
    output_dir = "./"
    # ----

    cache_dir="./"

    classifier = Big5PersonalityClassifier(
        model_name=model_name,
        cache_dir=cache_dir,
        output_dir=output_dir
    )

    if TRAIN_FLAG:
        classifier.run(evaluate_flag=False)
    else:
        classifier.run(evaluate_flag=True)