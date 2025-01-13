import copy
import functools
from multiprocessing import Pool

import datasets
import torch
from arclib.arc import read_tasks_from_single_file
from arclib.augmentations.utils import get_augmenters, process_task
from arclib.messagers import GPTTextMessageRepresenterV2
from arclib.representers import (
    PythonListGridRepresenter,
    TextExampleRepresenter,
    TextTaskRepresenter,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import os
import re

def get_latest_checkpoint(checkpoint_dir):
    # Find all checkpoint directories
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    
    # Extract numbers and find max
    checkpoint_numbers = [int(re.search(r'checkpoint-(\d+)', cp).group(1)) for cp in checkpoints]
    
    if not checkpoint_numbers:
        return None
        
    latest_number = max(checkpoint_numbers)
    return f"checkpoint-{latest_number}"

def evaluate_arc_task(model, tokenizer, task, formatter, max_length=600):
    """
    Simple evaluation function for ARC tasks.
    Returns both the model output and formatted inputs for analysis.
    """
    # Format the task examples
    formatted_examples = formatter.encode(task)
    results = []
    
    # Process each example
    for example in formatted_examples:
        # Format as chat
        prompt = tokenizer.apply_chat_template(example, tokenize=False)
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                #temperature=0.7,  # Adjust as needed
                #do_sample=True
            )
        
        # Decode
        generated = tokenizer.decode(
            outputs[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        results.append({
            'prompt': prompt,
            'generated': generated,
            'target': example[-1]['content'] if isinstance(example, list) else None
        })
    
    return results

arc_path = "./data/"
path = arc_path + "arc-agi_training_challenges.json"
solution_file = arc_path + "arc-agi_training_solutions.json"

arc_test_tasks = read_tasks_from_single_file(
    path, test=True, solution_file=solution_file
)

num_tasks = 5

arc_test_tasks = [task for task in arc_test_tasks if "-0" in task.name]
if num_tasks is not None:
    arc_test_tasks = arc_test_tasks[:num_tasks]


print(arc_test_tasks)
print("Number of train tasks: ", len(arc_test_tasks))

standard_formatter = TextTaskRepresenter(
    example_representer=TextExampleRepresenter(
        io_sep=" -> ",
        input_header="",
        output_header="",
        output_footer="#",
        grid_representer=PythonListGridRepresenter(),
    )
)
formatter = GPTTextMessageRepresenterV2(task_representer=standard_formatter)

# get model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
quantize = True
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={'':torch.cuda.current_device()}, #"auto",
    quantization_config=bnb_config if quantize else None,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
peft_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "modules_to_save": None,
}
training_config = {
    "bf16": True,
    "do_eval": False,
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "max_steps": -1,
    "output_dir": "./checkpoint_dir_ttt_",
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 2,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
    "report_to": "wandb",
    
}
train_conf = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)


# select augmentations
augmenters_to_apply = get_augmenters(
    include_basic=True, include_size=True, include_chain=True, include_repeat=True
)
processor = functools.partial(
    process_task,
    augmenters=augmenters_to_apply,
    formatter=formatter,
    tokenizer=tokenizer,
    permute_n=1,
    Nmax=250,
    seed=42,
)
cpus = 8
with Pool(cpus) as p:
    data = p.map(processor, arc_test_tasks)

assert len(data) == len(arc_test_tasks)


def apply_chat_template(example, tokenizer, add_generation_prompt=False):
    messages = copy.deepcopy(example["input"])
    messages.append({"role": "assistant", "content": example["output"]["content"]})

    # Use tokenizer to format the chat messages but don't tokenize yet
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Return raw text instead of tokenized output
        add_generation_prompt=add_generation_prompt
    )

    return {
        "text": text,
        "label": example["output"]["content"]
    }


task_outdir = train_conf.output_dir

for train_data, t in zip(data, arc_test_tasks):
    
    task_id = t.name.replace("-0", "")
    train_conf.output_dir = task_outdir + task_id
    
    # per task data
    d1 = datasets.Dataset.from_list(train_data)
    processed_train_dataset = d1.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=5,
        desc="Applying chat template to train_sft",
    )

    trainer = SFTTrainer(
        model=base_model,
        args=train_conf,
        peft_config=peft_conf,
        train_dataset=processed_train_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True,
        #predict_with_generate=True,
    )
    train_result = trainer.train()
    # metrics = train_result.metrics
    # trainer.log_metrics("train", metrics)
    #trainer.save_metrics("train", metrics)
    #trainer.save_state()
    # ############
    # # Save model
    # ############
    # trainer.save_model(train_conf.output_dir)


    # EVAL
#     formatted_test = formatter.encode(t)
#     test_data = datasets.Dataset.from_list([{"input":formatted_test[0], "output": formatted_test[1]}])
#     processed_test_dataset = test_data.map(
#             apply_chat_template,
#             fn_kwargs={"tokenizer": tokenizer},
#             num_proc=1,
#             desc="Applying chat template to test data",
#         ).map(
#     lambda example: {
#         "input_ids": tokenizer(
#             example["text"], truncation=True, padding="max_length", max_length=2048
#         )["input_ids"][0],
#         "attention_mask": tokenizer(
#             example["text"], truncation=True, padding="max_length", max_length=2048
#         )["attention_mask"][0],
#         "labels": tokenizer(
#             example["label"], truncation=True, padding="max_length", max_length=2048
#         )["input_ids"][0]
#     },
#     batched=True
# )
    checkpoint_dir = train_conf.output_dir
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        ft_model = PeftModel.from_pretrained(base_model, checkpoint_path)


    # Evaluate
    results = evaluate_arc_task(ft_model, tokenizer, t, formatter)
    print(results)