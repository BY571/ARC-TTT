"""
This module contains classes for representing tasks and examples as messages for chat-based interfaces.
"""

from abc import ABC, abstractmethod
from html import escape
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from arclib.arc import Example, Task
from arclib.representers import (
    CompositeRepresenter,
    #ConnectedComponentRepresenter,
    DiffExampleRepresenter,
    PythonListGridRepresenter,
    TaskRepresenter,
    TextTaskRepresenter,
)


MESSAGE = Dict[str, Union[str, Dict]]
MESSAGES = List[MESSAGE]


def display_messages(messages: MESSAGES):
    html_output = """<!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>Chat View</title>
    <style>
    /* CSS styling for chat interface */
    body {
    font-family: Arial, sans-serif;
    background-color: #f5f5f5;
    }
    .chat-container {
    width: 80%;
    max-width: 800px;
    margin: 0 auto;
    margin-top: 50px;
    }
    .message {
    display: block;
    clear: both;
    margin-bottom: 15px;
    }
    .message.user {
    text-align: right;
    }
    .message.assistant {
    text-align: left;
    }
    .message.system {
    text-align: left;
    }
    .message .bubble {
    display: inline-block;
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 70%;
    position: relative;
    }
    .message.user .bubble {
    background-color: #0084ff;
    color: white;
    }
    .message.assistant .bubble {
    background-color: #e5e5ea;
    color: black;
    }
    .message.system .bubble {
    background-color: #e5e5ea;
    color: black;
    }
    .message .bubble img {
    max-width: 100%;
    border-radius: 10px;
    }
    .message .role {
    font-size: 0.8em;
    color: black;
    margin-bottom: 5px;
    }
    </style>
    </head>
    <body>
    <div class="chat-container">
    """

    # Loop through messages
    for message in messages:
        role = message.get("role", "user")
        content_list = message.get("content", [])
        if not content_list:
            continue  # Skip if no content
        if isinstance(content_list, str):
            content_list = [{"type": "text", "text": content_list}]

        # Start message div
        html_output += f'<div class="message {role}">\n'
        # Start bubble div
        html_output += '<div class="bubble">\n'
        # Add role label inside the bubble
        html_output += f'<div class="role">{role.capitalize()}</div>\n'

        # Process content items
        for content in content_list:
            content_type = content.get("type")
            if content_type == "text":
                text = content.get("text", "")
                # Escape HTML entities in text
                safe_text = escape(text)
                # Replace newlines with <br>
                safe_text = safe_text.replace("\n", "<br>")
                html_output += f"<p>{safe_text}</p>\n"
            elif content_type == "image_url":
                image_url = content["image_url"].get("url", {})
                if image_url:
                    html_output += f'<img src="{image_url}" alt="Image">\n'
            else:
                # Handle other content types if necessary
                pass

        # Close bubble and message divs
        html_output += "</div>\n</div>\n"

    # Close chat-container and body tags
    html_output += """
</div>
</body>
</html>"""

    return html_output


class MessageRepresenter(ABC):
    task_representer: TaskRepresenter

    @abstractmethod
    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]:
        pass

    def display(self, messages: MESSAGES):
        return display_messages(messages)


# =============== MESSAGE REPRESENTATION ===============


class GPTTextMessagerepresenter(MessageRepresenter):
    def __init__(
        self,
        prompt: Optional[
            str
        ] = "Figure out the pattern in the following examples and apply it to the test case. {description}Your answer must follow the format of the examples. \n",
        task_representer: TaskRepresenter = TextTaskRepresenter(),
    ):
        self.prompt = prompt
        self.task_representer = task_representer

    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        if hasattr(task, "description"):
            desciption = "Here is a description of the task: \n\n{description}\n"
            description = desciption.format(description=task.description)
            prompt = self.prompt.format(description=description)
        else:
            prompt = self.prompt.format(description="")

        input_data.append({"role": "system", "content": prompt})

        for example in task.train_examples:
            query, output = self.task_representer.example_representer.encode(
                example, **kwargs
            )
            input_data.append({"role": "system", "content": query + output})

        query, output = self.task_representer.example_representer.encode(
            task.test_example, **kwargs
        )

        input_data.append({"role": "user", "content": query})

        output_data = {"role": "assistant", "content": output}

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        raise NotImplementedError(
            "Decoding for GPTTextMessagerepresenter is not implemented."
        )


class GPTTextMessageRepresenterV2(MessageRepresenter):
    def __init__(
        self,
        prompt: Optional[
            str
        ] = "Figure out the underlying transformation in the following examples and apply it to the test case. {description}Here are some examples from this transformation, your answer must follow the format.\n",
        task_representer: TaskRepresenter = TextTaskRepresenter(),
    ):
        self.prompt = prompt
        self.task_representer = task_representer

    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]:
        input_data = []

        if hasattr(task, "description"):
            description = task.description
            description = f"\n\n A possible description of the transformation: \n\n{description}\n"
            prompt = self.prompt.format(description=description)
        else:
            prompt = self.prompt.format(description="")

        if isinstance(
            self.task_representer.example_representer, DiffExampleRepresenter
        ):
            if self.task_representer.example_representer.use_output:
                prompt += "The input-diff-output grids are provided as python arrays where the diff is simply the output minus input:\n"
            else:
                prompt += "The input-diff grids are provided as python arrays:\n"
        # elif isinstance(
        #     self.task_representer.example_representer.grid_representer,
        #     ConnectedComponentRepresenter,
        # ):
        #     connected_component = kwargs.get(
        #         "connected_component",
        #         self.task_representer.example_representer.grid_representer.connected_component,
        #     )
        #     connected_component = (
        #         "including diagonals"
        #         if connected_component == 8
        #         else "excluding diagonals"
        #     )
        #     prompt += f"The input-output grids are provided with indices of connected shapes ({connected_component}) of the same color:\n"
        elif isinstance(
            self.task_representer.example_representer.grid_representer,
            PythonListGridRepresenter,
        ):
            prompt += "The input-output grids are provided as python arrays:\n"
        elif isinstance(
            self.task_representer.example_representer.grid_representer,
            CompositeRepresenter,
        ):
            connected_component = kwargs.get(
                "connected_component",
                self.task_representer.example_representer.grid_representer.connected_component,
            )
            connected_component = (
                "including diagonals"
                if connected_component == 8
                else "excluding diagonals"
            )
            prompt += f"The input-output grids are provided as both python arrays and indices of connected shapes ({connected_component}) of the same color:\n"

        for example in task.train_examples:
            query, output = self.task_representer.example_representer.encode(
                example, **kwargs
            )
            if query is None or output is None:
                return None, None
            prompt += query + output + "\n"

        input_data.append({"role": "system", "content": prompt})

        query, output = self.task_representer.example_representer.encode(
            task.test_example, **kwargs
        )
        if query is None or output is None:
            return None, None

        input_data.append({"role": "user", "content": query})

        output_data = {"role": "assistant", "content": output}

        return input_data, output_data

    def decode(self, input_data: MESSAGES, output_data: MESSAGE, **kwargs) -> Task:
        raise NotImplementedError(
            "Decoding for GPTTextMessageRepresenterV2 is not implemented."
        )

    def __repr__(self) -> str:
        return f"GPTTextMessageRepresenterV2(prompt={self.prompt!r}, task_representer={repr(self.task_representer)})"
