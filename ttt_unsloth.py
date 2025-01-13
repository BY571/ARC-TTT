import copy
import functools
import os
from datetime import datetime
from multiprocessing import Pool

import datasets
from arclib.arc import read_tasks_from_single_file
from arclib.augmentations.utils import get_augmenters, process_task
from arclib.messagers import GPTTextMessageRepresenterV2
from arclib.representers import (
    PythonListGridRepresenter,
    TextExampleRepresenter,
    TextTaskRepresenter,
)
from transformers import DataCollatorForSeq2Seq, TrainingArguments

from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only


def compare_and_score_strings(label, pred):
    try:
        length = len(label)
        if length < len(pred):
            pred = pred[:length]
        elif length > len(pred):
            pred += " " * (length - len(pred))

        # Initialize the score
        matching_characters = sum(1 for i in range(length) if label[i] == pred[i])

        # Check if the strings are exactly the same
        all_match = label == pred

        return all_match, matching_characters / length
    except Exception as e:
        print(e)
        return False, 0.0


def extract_assistant_output(text):
    """
    Extracts the assistant's output from the given text format.

    Args:
        text (str or list): The input text containing the full conversation

    Returns:
        str: The extracted assistant's output
    """
    # If input is a list, take the first element
    if isinstance(text, list):
        text = text[0]

    # Find the assistant's section
    assistant_start = text.find("<|start_header_id|>assistant<|end_header_id|>")
    if assistant_start == -1:
        return None

    # Find the end of assistant's output
    assistant_end = text.find("<|eot_id|>", assistant_start)
    if assistant_end == -1:
        return None

    # Extract the content between the tags
    assistant_content = text[assistant_start:assistant_end]

    # Remove the header tags
    content_start = assistant_content.find("<|end_header_id|>") + len(
        "<|end_header_id|>"
    )
    output = assistant_content[content_start:].strip()

    return output


def get_model(base_model):
    return FastLanguageModel.get_peft_model(
        base_model,
        r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )


def formatting_prompts_func(examples):
    messages = copy.deepcopy(examples["input"])
    messages.append({"role": "assistant", "content": examples["output"]["content"]})
    texts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {
        "text": texts,
    }


arc_path = "./data/"
path = arc_path + "arc-agi_training_challenges.json"
solution_file = arc_path + "arc-agi_training_solutions.json"

arc_test_tasks = read_tasks_from_single_file(
    path, test=True, solution_file=solution_file
)

num_tasks = 20

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
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
quantize = True
max_seq_length = 5000  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)


# select augmentations
augmenters_to_apply = get_augmenters(
    include_basic=True, include_size=True, include_chain=True, include_repeat=True
)
processor = functools.partial(
    process_task,
    augmenters=augmenters_to_apply,
    formatter=formatter,
    permute_n=1,
    Nmax=250,
    seed=42,
)
cpus = 8
with Pool(cpus) as p:
    aug_data = p.map(processor, arc_test_tasks)

assert len(aug_data) == len(arc_test_tasks)


# define experiment outdir

current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
outdir = f"./output_{current_date}/"
filename = f"eval_results.txt"

for train_data, t in zip(aug_data, arc_test_tasks):

    task_id = t.name.replace("-0", "")
    task_outdir = outdir + task_id

    # per task data
    train_data = datasets.Dataset.from_list(train_data)
    processed_train_dataset = train_data.map(formatting_prompts_func)

    ft_model = get_model(copy.deepcopy(base_model))
    trainer = SFTTrainer(
        model=ft_model,
        tokenizer=tokenizer,
        train_dataset=processed_train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            num_train_epochs=2,  # Set this for 1 full training run.
            # max_steps=20,
            learning_rate=1e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=task_outdir,
            report_to="wandb",
        ),
    )
    # We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    trainer_stats = trainer.train()

    # # Evaluate
    FastLanguageModel.for_inference(ft_model)  # Enable native 2x faster inference

    test_input, test_output = formatter.encode(t)
    test_input = tokenizer.apply_chat_template(
        test_input,
        tokenize=True,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
    ).to("cuda")
    outputs = ft_model.generate(
        input_ids=test_input,
        max_new_tokens=2000,
        use_cache=True,
        temperature=0.1,
        min_p=0.1,
    )
    full_test_outputs_pred = tokenizer.batch_decode(outputs)
    test_output_pred = extract_assistant_output(full_test_outputs_pred)
    correct, score = compare_and_score_strings(test_output["content"], test_output_pred)
    print(
        "Task: ",
        task_id,
        "Correct: ",
        correct,
        "Score: ",
        score,
    )
    # model.save_pretrained("lora_model")  # Local saving
    # tokenizer.save_pretrained("lora_model")
    # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
    # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving
    eval_logs = {"correct": float(correct), "score": score}
    trainer.log(eval_logs)
    # write to local txt file
    filepath = os.path.join(outdir, filename)

    # Check if the file exists
    if not os.path.exists(filepath):
        # File does not exist: Create it and write the first entry
        with open(filepath, "w") as f:
            f.write(f"Task: {task_id}, Correct: {correct}, Score: {score}\n")
    else:
        # File exists: Append the new entry
        with open(filepath, "a") as f:
            f.write(f"Task: {task_id}, Correct: {correct}, Score: {score}\n")

    # write eval solution
    with open(os.path.join(task_outdir, f"eval_{task_id}.txt"), "w") as f:
        f.write(
            "SOLUTION:\n\n"
            + test_output["content"]
            + "\n\nPREDICTION:\n\n"
            + test_output_pred
        )

    del ft_model
