import os
import json
import time
import torch
import evaluate
import numpy as np
import seaborn as sns
from tqdm import tqdm
from datasets import Value
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding


class SentimentClassifier:
    """
    A sentiment classification model based on RoBERTa.
    
    This class provides an end-to-end pipeline for training, evaluating,
    and visualizing a sentiment classifier.
    """

    def __init__(self, base_model="roberta-base", num_labels=2):
        self.base_model = base_model
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        self.model_dir = None

    def prepare_data(self, train_data, dev_data, max_length=512):
        if self.tokenizer is None:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(self.base_model)

        def tokenize_function(examples):
            return self.tokenizer(
                examples["input"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        self.train_data = train_data.map(tokenize_function, batched=True, desc="Tokenizing training data")
        self.dev_data = dev_data.cast_column('label', Value('int64')).map(
            tokenize_function, batched=True, desc="Tokenizing validation data"
        )

        return self

    def train(self, output_dir, experiment_name=None, epochs=5, lr=2e-5,
              batch_size=2, seed=42, timestamp=None, report_to="none"):
        """Fine-tune the classifier. ``report_to`` is passed to ``TrainingArguments``
        (e.g. ``"wandb"``); the default ``"none"`` disables experiment tracking."""
        if timestamp is None:
            timestamp = int(time.time())
        if experiment_name is None:
            experiment_name = "sentiment-classifier"

        self.model = RobertaForSequenceClassification.from_pretrained(
            self.base_model, num_labels=self.num_labels
        )

        accuracy = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)

        data_collator = DataCollatorWithPadding(self.tokenizer)
        self.model_dir = os.path.join(output_dir, f"{experiment_name}-{timestamp}")
        run_name = f"{experiment_name}-{epochs}epochs-{timestamp}"

        training_args = TrainingArguments(
            output_dir=self.model_dir,
            overwrite_output_dir=True,
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            num_train_epochs=epochs,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=0.1,
            weight_decay=0.01,
            seed=seed,
            run_name=run_name,
            report_to=report_to,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.train_data,
            eval_dataset=self.dev_data,
            compute_metrics=compute_metrics,
            processing_class=self.tokenizer,
        )

        self.trainer.train()
        return self

    def load_model(self, model_dir=None):
        if model_dir is None:
            model_dir = self.model_dir

        self.model = RobertaForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
        return self

    def evaluate(self, test_data, num_samples=None, threshold=0.5,
                 positive_class=1, output_dir=None):
        results = self._compute_metrics(
            test_data, 
            self.model, 
            self.tokenizer, 
            num_samples, 
            threshold, 
            positive_class, 
            output_dir
        )
        self.results = results
        return results

    def _compute_metrics(self, dataset, model, tokenizer, num_samples=None,
                         threshold=0.5, positive_class=1, save_dir=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        if num_samples is None:
            num_samples = len(dataset)
        else:
            num_samples = min(num_samples, len(dataset))

        y_true, y_pred, y_scores = [], [], []

        for i in tqdm(range(num_samples), desc="Computing metrics"):
            text = dataset[i]['input']
            true_label = dataset[i]['label']

            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=1).detach().cpu().numpy()[0]
            score = probs[positive_class]

            predicted_label = int(score >= threshold)

            y_true.append(true_label)
            y_pred.append(predicted_label)
            y_scores.append(score)

        y_true, y_pred, y_scores = map(np.array, [y_true, y_pred, y_scores])

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = np.mean(y_pred == y_true)
        f1 = f1_score(y_true, y_pred)

        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)

        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'roc_data': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist(),
                'auc': roc_auc
            },
            'pr_data': {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'thresholds': pr_thresholds.tolist(),
                'average_precision': avg_precision
            },
            'predictions': {
                'true': y_true.tolist(),
                'pred': y_pred.tolist(),
                'scores': y_scores.tolist()
            }
        }

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            summary = {
                "Evaluation Samples": num_samples,
                "Accuracy": accuracy,
                "F1 Score": f1,
                "Precision": precision,
                "Recall": recall,
                "Specificity": specificity,
                "AUC-ROC": roc_auc,
                "Average Precision": avg_precision
            }
            with open(os.path.join(save_dir, 'metrics_results.json'), 'w') as f:
                json.dump(summary, f, indent=4)
            print(f"Metrics saved to {os.path.join(save_dir, 'metrics_results.json')}")

        return results

    def visualize(self, output_dir=None, results=None):
        if results is None:
            results = self.results

        if output_dir is None:
            output_dir = os.path.join(self.model_dir, "results")

        os.makedirs(output_dir, exist_ok=True)

        self._plot_roc_curve(results, save_dir=output_dir)
        self._plot_precision_recall_curve(results, save_dir=output_dir)
        self._plot_confusion_matrix(results, save_dir=output_dir)

        return self

    def _plot_roc_curve(self, results, save_dir=None):
        fpr, tpr, auc_score = results['roc_data']['fpr'], results['roc_data']['tpr'], results['roc_data']['auc']
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
        plt.close()

    def _plot_precision_recall_curve(self, results, save_dir=None):
        precision = results['pr_data']['precision']
        recall = results['pr_data']['recall']
        avg_precision = results['pr_data']['average_precision']
        plt.figure()
        plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'))
        plt.close()

    def _plot_confusion_matrix(self, results, save_dir=None):
        cm = results['confusion_matrix']
        matrix = np.array([[cm['true_negative'], cm['false_positive']],
                           [cm['false_negative'], cm['true_positive']]])
        plt.figure()
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
