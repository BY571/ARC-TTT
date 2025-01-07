import json
import math
import random
import re
from collections import Counter


def read_json(filename):
    with open(filename, "r", encoding="utf-8") as file:
        content = json.load(file)
    return content


def tokenize_string(string, possible_token):
    # Create a regular expression pattern based on the tokens
    pattern = "|".join(re.escape(token) for token in possible_token)

    # Use re.findall to extract all matches
    result = re.findall(pattern, string)
    return result


def format_matrix_as_spreadsheet(matrix):
    # Create column headers (A, B, ..., Z, AA, AB, ..., AD)
    columns = [chr(i) for i in range(65, 91)]  # A-Z
    columns += ["A" + chr(i) for i in range(65, 69)]  # AA-AD

    # Prepare the header row
    header = "    " + " ".join(columns[: len(matrix[0])]) + "\n"

    # Prepare the body with row labels and content
    body = ""
    for i, row in enumerate(matrix, 1):
        row_str = f"{i:2}  " + " ".join(f"{num}" for num in row) + "\n"
        body += row_str

    # Combine header and body
    formatted_string = header + body
    return formatted_string


def matrix_to_string(matrix):
    return "\n".join([" ".join(map(str, row)) for row in matrix])


def format_task(
    tasks,
    random_examples=False,
    preprompt="",
    test_prompt="\n\nThe task for you to solve:\nInput\n",
    include_final_output=True,
):

    final_output = preprompt

    if random_examples:
        random.shuffle(tasks)

    for i, task in enumerate(tasks[:-2]):
        final_output += format_example(i, task["input"], task["output"])

    final_output += test_prompt + matrix_to_string(tasks[-1]["input"]) + "\n"
    if include_final_output:
        final_output += "Output\n"

    output = matrix_to_string(tasks[-1]["output"])

    return final_output, output


def format_example(i, input, output):
    return (
        f"\n\nExample {i}\nInput"
        + "\n"
        + matrix_to_string(input)
        + "\n"
        + f"Output"
        + "\n"
        + matrix_to_string(output)
        + "\n"
    )
