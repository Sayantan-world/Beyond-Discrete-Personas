import pandas as pd
from nltk.translate.meteor_score import meteor_score
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from pathlib import Path
import argparse
from tqdm.auto import tqdm
from evaluate import load
import os

class EvaluationMetrics:
    def __init__(self, input_file, output_file, cache_dir):
        self.input_file = input_file
        self.output_file = output_file
        self.cache_dir = cache_dir
        self.df = None
        self.load_data()

        # Set environment variables for Hugging Face cache
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_data(self):
        """Load the JSON file containing predictions and references."""
        self.df = pd.read_json(self.input_file)
        print(f"Data loaded successfully from {self.input_file}")

    def calculate_bleu_score(self, references, candidates):
        """Calculate BLEU score."""
        smoothing = SmoothingFunction().method4
        bleu_scores = [
            sentence_bleu(
                [ref], cand, smoothing_function=smoothing, weights=(0.25, 0.25, 0.25, 0.25)
            )
            for ref, cand in zip(references, candidates)
        ]
        return np.mean(bleu_scores)

    def calc_meteor(self, references, candidates):
        """Calculate METEOR score."""
        meteor = load('meteor', cache_dir=self.cache_dir)
        n = len(references)
        results = []
        for i in range(n):
            result = meteor.compute(predictions=[candidates[i]], references=[references[i]])
            results.append(result['meteor'])
        results = sorted(results, reverse=True)
        return sum(results) / len(results)

    def calc_bertscore(self, references, candidates):
        """Calculate BERTScore."""
        bertscore = load("bertscore", cache_dir=self.cache_dir)
        results = bertscore.compute(predictions=candidates, references=references, model_type="bert-base-uncased", device=self.device)
        sorted_bertsc_f1 = sorted(results["f1"], reverse=True)
        return sum(sorted_bertsc_f1) / len(sorted_bertsc_f1)

    def calc_rougescore(self, references, candidates):
        """Calculate ROUGE score."""
        rouge = load('rouge', cache_dir=self.cache_dir)
        return rouge.compute(predictions=candidates, references=references, rouge_types=["rouge1", "rouge2", "rougeL"])

    def run_evaluation(self):
        """Run the evaluation metrics and save the results to a text file."""

        self.df = self.df[self.df['golden'].str.strip().astype(bool) & self.df['prediction'].str.strip().astype(bool)]

        references = list(self.df['golden'])
        candidates = list(self.df['prediction'])

        bleu_score = self.calculate_bleu_score(references, candidates)
        meteor_score = self.calc_meteor(references, candidates)
        bert_score = self.calc_bertscore(references, candidates)
        rouge_score = self.calc_rougescore(references, candidates)

        with open(self.output_file, 'w') as file:
            file.write(f"BLEU Score: {bleu_score}\n")
            file.write(f"METEOR Score: {meteor_score}\n")
            file.write(f"BERT Score: {bert_score}\n")
            file.write(f"ROUGE Score: {rouge_score}\n")

        print(f"Evaluation results saved to {self.output_file}")

    @staticmethod
    def parse_arguments():
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(description="Evaluate dialogue generation using various metrics.")
        parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file.")
        parser.add_argument('--output_file', type=str, required=True, help="Path to the output text file.")
        parser.add_argument('--cache_dir', type=str, required=True, help="Path to the cache directory.")
        return parser.parse_args()

    @classmethod
    def from_arguments(cls):
        """Create an instance of the class using command-line arguments."""
        args = cls.parse_arguments()
        return cls(args.input_file, args.output_file, args.cache_dir)

if __name__ == '__main__':
    evaluator = EvaluationMetrics.from_arguments()
    evaluator.run_evaluation()
