from format_utils import read_json, format_task
from itertools import islice
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
import torch
import numpy as np
import warnings

def create_training_dataset(data, tokenizer, max_length=512):
    """Create a dataset from the puzzle data."""
    training_examples = []
    sequence_lengths = []
    num_truncated = 0
    
    for task_name, task in data.items():
        # Create the prompt structure
        sysprompt = """You are a world-class puzzle solver with exceptional pattern recognition skills. Your task is to analyze puzzles, spot patterns, and provide direct solutions."""
        
        preprompt = """Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines.\nHere are the input and output grids for the reference examples:\n"""
        outprompt = """Directly provide the output grids corresponding to the given test input grids, based on the patterns observed in the reference examples."""
        
        # Format the task
        examples, output = format_task(task["train"], test_prompt="\n\nHere is the input grid for the test example:\n", include_final_output=False)
        
        # Combine all prompts
        input_data = [
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": preprompt + examples + outprompt}
        ]
        
        output = f"The output grid for the test input grid is:\n\n```\n{output}\n```"
        output_data = {"role": "assistant", "content": output}
        
        messages = input_data + [output_data]
        
        # Tokenize the entire conversation
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        original_length = encoded.shape[1]
        sequence_lengths.append(original_length)
        
        # Check if truncation is needed and warn if it occurs
        if original_length > max_length:
            warnings.warn(f"Conversation for task '{task_name}' exceeded max_length ({original_length} > {max_length} tokens). Truncating sequence. Consider increasing max_length if this is undesirable.")
            encoded = encoded[:, :max_length]
            num_truncated += 1
            
        training_examples.append({
            "input_ids": encoded.squeeze().numpy(),
            "attention_mask": np.ones_like(encoded.squeeze().numpy())
        })
    
    # Print sequence length statistics
    print("\nSequence Length Statistics:")
    print(f"Maximum sequence length: {max(sequence_lengths)} tokens")
    print(f"Average sequence length: {sum(sequence_lengths)/len(sequence_lengths):.2f} tokens")
    print(f"Number of truncated sequences: {num_truncated} out of {len(sequence_lengths)} ({(num_truncated/len(sequence_lengths)*100):.1f}%)")
    
    if num_truncated > 0:
        print("\nWarning: Some sequences were truncated. Consider increasing max_length if this is undesirable.")
    
    return Dataset.from_list(training_examples)

def train_model(model, train_dataset, output_dir, training_args=None):
    """Fine-tune the model with the given dataset."""
    if training_args is None:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=100,
            learning_rate=2e-5,
            fp16=True if torch.cuda.is_available() else False,
            save_total_limit=2,
        )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()
    
    # Save the final model
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    

    # Load and prep dataset
    data = read_json("data/arc-agi_training_challenges.json")
    data = dict(islice(data.items(), 0, 200))  # Slice for testing
    
    # Initialize model and tokenizer
    checkpoint = "microsoft/Phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map="auto",
        #quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Create dataset
    train_dataset = create_training_dataset(data, tokenizer, max_length=5000)
    
    # Train the model
    output_dir = "puzzle_solver_model"
    train_model(model, train_dataset, output_dir)