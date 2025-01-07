import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple

class ARCEvalDataset(Dataset):
    def __init__(self, data: Dict[str, Dict], solutions: Dict[str, List], tokenizer=None, prompt_template: str = None):
        self.data = []
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template or self.default_prompt_template
        input_lengths = []
        for sample_name, sample_data in data.items():
            train_examples = sample_data['train']
            test_input = sample_data['test'][0]['input']
            solution = solutions[sample_name]

            # Prepare input text
            input_text = self.prepare_input_text(train_examples, test_input)

            # If tokenizer is not none we tokenize
            if self.tokenizer:
                tokenized_input = self.tokenizer.encode(input_text, add_special_tokens=True)
                input_lengths.append(len(tokenized_input))
                self.data.append({
                    'sample_name': sample_name,
                    'raw_input': input_text,
                    'raw_test_input': test_input,
                    'tokenized_input': torch.tensor([tokenized_input]),
                    'solution': solution
                })
            else:
                self.data.append({
                    'sample_name': sample_name,
                    'raw_input': input_text,
                    'raw_test_input': test_input,
                    'solution': solution
                })
        if self.tokenizer:
            print("Dataset size: ", len(input_lengths))
            print("Maximum input token: ", max(input_lengths))
            print("Minimum input token: ", min(input_lengths))
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def default_prompt_template(self):
        return """You are given {num_examples} examples and you are supposed to solve the test example I will provide you. Here are the examples to help you find the solution:

{examples}

And here is your test example:

{test_input}

Write a python function called 'transform_grid' to solve the problem. Use chain of thought reasoning!
"""

    def prepare_input_text(self, train_examples: List[Dict], test_input: List[List[int]]) -> str:
        num_examples = len(train_examples)
        
        examples_text = "\n\n".join([
            f"Example {i+1}:\nInput: {example['input']}\nOutput: {example['output']}"
            for i, example in enumerate(train_examples)
        ])
        
        test_input_text = f"Input: {test_input}"
        
        return self.prompt_template.format(
            num_examples=num_examples,
            examples=examples_text,
            test_input=test_input_text
        )