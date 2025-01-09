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

def evaluate_adapter(
    trainer,
    test_task,
    formatter,
    tokenizer,
    device="cuda",
    max_new_tokens=500,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
):
    """
    Evaluate a trained adapter on a test task.
    
    Args:
        trainer: SFTTrainer instance with trained model
        test_task: ARC task to evaluate on
        formatter: Task formatter instance
        tokenizer: Tokenizer instance
        device: Device to run inference on
        max_new_tokens: Maximum number of tokens to generate
        num_return_sequences: Number of sequences to generate per input
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    
    Returns:
        dict: Dictionary containing evaluation results
    """
    # Put model in evaluation mode
    trainer.model.eval()
    
    # Format test task
    formatted_test = formatter.encode(test_task)
    results = []
    
    # Evaluate on each test example
    for test_example in formatted_test:
        # Prepare input
        inputs = tokenizer.apply_chat_template(
            test_example, 
            return_tensors="pt"
        ).to(device)
        
        # Generate with specified parameters
        with torch.no_grad():
            outputs = trainer.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True if temperature > 0 else False,
            )
        
        # Process outputs
        input_length = len(inputs[0])
        generated_responses = tokenizer.batch_decode(
            outputs[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Store results
        results.append({
            "input": tokenizer.decode(inputs[0], skip_special_tokens=True),
            "generated_responses": generated_responses,
            "ground_truth": test_example[-1]["content"]  # Assuming last message is ground truth
        })
    
    return {
        "task_name": test_task.name,
        "results": results
    }


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
    device_map="auto",
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


def apply_chat_template(
    example,
    tokenizer,
):
    # list of all answer message attempts
    messages = copy.copy(example["input"])
    messages.append(example["output"])
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return example

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
    # model = PeftModel.from_pretrained(model, adapter_model_name)
    # generate test task prompt:
    # test_task = formatter.encode(arc_test_tasks[0])
    # inputs = tokenizer.apply_chat_template(test_task[0], return_tensors="pt")
    # outputs = trainer.model.generate(inputs, max_new_tokens=500)
    # input_length = len(inputs[0])
    # print("\n\n", tokenizer.batch_decode(inputs[0]))
    # outtext = tokenizer.batch_decode(outputs[:, input_length:])[0]
    # print("\n\n", outtext)

    # Evaluation
    print(f"\nEvaluating task: {task_id}")
    eval_results = evaluate_adapter(
        trainer=trainer,
        test_task=t,
        formatter=formatter,
        tokenizer=tokenizer
    )
    # Print results
    for idx, result in enumerate(eval_results["results"]):
        print(f"\nTest Example {idx + 1}")
        print("-" * 50)
        print(f"Input:\n{result['input']}\n")
        print("Generated Responses:")
        for i, resp in enumerate(result["generated_responses"], 1):
            print(f"{i}. {resp}")
        print(f"\nGround Truth:\n{result['ground_truth']}")
        print("-" * 50)
