import os
import json
import torch
import argparse
import logging
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel, PeftConfig

# Set logging level to ERROR to suppress unwanted warnings
logging.basicConfig(level=logging.ERROR)

class ZeroShotInference:
    def __init__(self, model_name, input_file, output_file, cache_dir, hf_token, adapter_path, seed=42):
        self.model_name = model_name
        self.input_file = input_file
        self.output_file = output_file
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.adapter_path = adapter_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = seed
        set_seed(self.seed)
        # Set environment variables
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            token=hf_token,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.config = PeftConfig.from_pretrained(adapter_path)
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path, device_map="auto")

    def load_data(self):
        """Load the JSON file containing dialogues."""
        with open(self.input_file, 'r') as file:
            self.dialogues = json.load(file)
        print(f"Data loaded successfully from {self.input_file}")

    def create_chats(self, dialogue):
        """Create chat history from dialogue."""
        chats = []
        for i in range(8):
            chat = []
            for j in range(2 * i + 1):
                role = "user" if j % 2 == 0 else "assistant"
                chat.append({"role": role, "content": dialogue[j]})
            chats.append(chat)
        return chats

    def generate_responses(self, chats):
        """Generate responses using the model."""
        instruction = "Reply concisely within 20 words "
        prompts = [(self.tokenizer.apply_chat_template(chat, tokenize=False)) for chat in chats]
        prompts = [instruction + prompt for prompt in prompts]

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        
        # Set the seed for reproducibility
        torch.manual_seed(self.seed)
        
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=256, do_sample=True)
        output = self.tokenizer.batch_decode(generated_ids)
        return output, prompts

    def post_process_output(self, output, input_prompts):
        """Post-process the generated output."""
        processed_outputs = []
        for response, prompt in zip(output, input_prompts):

            segments = response.split('assistant<|end_header_id|>\n\n')
            response = [segment for segment in segments if segment][-1]
            response = response.replace('<|end_of_text|>', '')
            response = response.replace('<|eot_id|>', '')
            response = response.strip()
            response = response.replace(prompt, '')

            if '<|begin_of_text|>' in response:
                response = response.split('\n')
                response = response[-1]
            # response = response.split('\n')
            # response = response[-1]
            processed_outputs.append(response)
        return processed_outputs

    def create_dialogue_dataset(self):
        """Create the dialogue dataset and save it to a JSON file."""
        results = []
        for dialogue in tqdm(self.dialogues, desc="Generating dialogues"):
            try:
                chats = self.create_chats(dialogue["dialogue"])
                responses, prompts = self.generate_responses(chats)
                predictions = self.post_process_output(responses, prompts)
                
                for i in range(8):
                    result = {
                        "author_fullname1": dialogue["author_fullname1"],
                        "author_fullname2": dialogue["author_fullname2"],
                        "author1": dialogue["author1"],
                        "author2": dialogue["author2"],
                        "id1": dialogue["id1"],
                        "id2": dialogue["id2"],
                        "prediction": predictions[i],
                        "golden": dialogue["dialogue"][2 * i + 1]
                    }
                    results.append(result)
            except Exception as e:
                print(f"Error processing dialogue: {e}")
            break

        with open(self.output_file, 'w') as file:
            json.dump(results, file, indent=2)
        print(f"Dialogue predictions created successfully and saved to {self.output_file}")

    @staticmethod
    def parse_arguments():
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(description="Run zero-shot inference with Mistral 7B.")
        parser.add_argument('--model_name', type=str, required=True, help="Model name or path.")
        parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file.")
        parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSON file.")
        parser.add_argument('--cache_dir', type=str, required=True, help="Path to the cache directory.")
        parser.add_argument('--hf_token', type=str, required=True, help="Hugging Face token.")
        parser.add_argument('--adapter_path', type=str, required=True, help="Path to adapter weights")
        parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
        return parser.parse_args()

    @classmethod
    def from_arguments(cls):
        """Create an instance of the class using command-line arguments."""
        args = cls.parse_arguments()
        return cls(args.model_name, args.input_file, args.output_file, args.cache_dir, args.hf_token, args.adapter_path, args.seed)

    def run(self):
        """Run the entire process."""
        self.load_data()
        self.create_dialogue_dataset()

if __name__ == '__main__':
    inference = ZeroShotInference.from_arguments()
    inference.run()
