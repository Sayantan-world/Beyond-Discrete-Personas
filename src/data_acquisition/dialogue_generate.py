import pandas as pd
import random
import json
import os
import time
import argparse
from pathlib import Path
from groq import Groq
from tqdm.auto import tqdm

# Set the API key as an environment variable
os.environ['GROQ_API_KEY'] = 'YOUR-API-KEY'

class DialogueGenerator:
    def __init__(self, input_file, output_dir, model="", seed=42):
        self.input_file = input_file
        self.output_dir = output_dir
        self.model = model
        self.seed = seed
        self.data = None
        self.client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
        self.errors = []  # To store any errors encountered during processing

    def load_data(self):
        """Load the JSON file into a pandas DataFrame."""
        with open(self.input_file, 'r') as f:
            self.data = json.load(f)
        print("Data loaded successfully")

    def generate_dialogue(self, journal_entry1, journal_entry2):
        """Generate dialogue using the API."""

        newline = "\n"
        instructions = f"Create a 9-turn dialogue in english between two authors based on the journal entries provided below. The dialogue should reflect a natural and engaging conversation, finding common ground between the authors' experiences, thoughts, or emotions. Ensure that the conversation stays true to the personality traits and tones expressed in the journal entries. Each author should contribute equally, with utterances that are concise, relevant, and no longer than 20 words.{newline}{newline} \
            Journal 1:{newline}{journal_entry1}{newline}{newline}Journal 2:{newline}{journal_entry2}"

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user", 
                    "content": instructions
                },
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        dialogue = completion.choices[0].message.content
        print(dialogue)
        completion_tokens = completion.usage.completion_tokens
        prompt_tokens = completion.usage.prompt_tokens

        return dialogue, completion_tokens, prompt_tokens

    def create_dialogue_dataset(self):
        """Create the dialogue dataset and save it to multiple JSON files with 1000 entries each."""
        dialogue_data = []
        file_counter = 1
        batch_size = 1000

        for i, entry in enumerate(tqdm(self.data, desc="Generating dialogues")):
            author_1 = entry['author_fullname1']
            author_2 = entry['author_fullname2']
            id1 = entry['id1']
            id2 = entry['id2']
            text1 = entry['journal_entry1']
            text2 = entry['journal_entry2']

            success = False
            while not success:
                try:
                    dialogue, completion_tokens, prompt_tokens = self.generate_dialogue(text1, text2)
                    dialogue_entry = {
                        "author_fullname1": author_1,
                        "author_fullname2": author_2,
                        "author1": entry['author1'],
                        "author2": entry['author2'],
                        "id1": id1,
                        "id2": id2,
                        "journal_entry1": text1,
                        "journal_entry2": text2,
                        "dialogue": dialogue,
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens
                    }
                    dialogue_data.append(dialogue_entry)
                    success = True
                except Exception as e:
                    error_entry = {
                        "author_fullname1": author_1,
                        "author_fullname2": author_2,
                        "author1": entry['author1'],
                        "author2": entry['author2'],
                        "id1": id1,
                        "id2": id2,
                        "error_message": str(e)
                    }
                    self.errors.append(error_entry)
                    time.sleep(5)
                # break
            # break

            time.sleep(5)  # Sleep for 10 seconds between API calls

            # if i==0:
            #     print(dialogue)
            # break

            # Save the data in batches of 1000 entries
            if (i + 1) % batch_size == 0 or (i + 1) == len(self.data):
                output_file = os.path.join(self.output_dir, f"dialogue_{file_counter:03}.json")
                with open(output_file, 'w') as outfile:
                    json.dump(dialogue_data, outfile, indent=4)
                print(f"Batch {file_counter} saved with {len(dialogue_data)} dialogues.")
                file_counter += 1
                dialogue_data = []  # Reset the batch

        # Save the errors to a file
        if self.errors:
            error_file = os.path.join(self.output_dir, "error.json")
            with open(error_file, 'w') as outfile:
                json.dump(self.errors, outfile, indent=4)
            print(f"Errors encountered and saved to {error_file}")

    @staticmethod
    def parse_arguments():
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(description="Generate dialogues from journal entries using an API.")
        parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file.")
        parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output JSON files.")
        parser.add_argument('--model', type=str, default="llama3-70b-8192", help="Model to use for dialogue generation.")
        parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
        return parser.parse_args()

    @classmethod
    def from_arguments(cls):
        """Create an instance of the class using command-line arguments."""
        args = cls.parse_arguments()
        return cls(args.input_file, args.output_dir, args.model, args.seed)

    def run(self):
        """Run the process to generate the dialogue dataset."""
        self.load_data()
        self.create_dialogue_dataset()


if __name__ == '__main__':
    generator = DialogueGenerator.from_arguments()
    generator.run()
