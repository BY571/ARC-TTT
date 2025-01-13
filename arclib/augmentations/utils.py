from typing import List
import itertools
import copy

import numpy as np
from arclib.arc import Task
from arclib.augmentations.augmenter import (
    Augmenter,
    Chain,
    Concat,
    Flip,
    IdentityAugmenter,
    IncreaseHeight,
    IncreaseResolution,
    IncreaseWidth,
    PermuteColors,
    PermuteExamples,
    RandomTranslateXY,
    Reflect,
    Repeat,
    Rotate,
    Transpose,
)
from arclib.messagers import MessageRepresenter #arc_to_messages

def get_augmenters(
    include_basic: bool = True,
    include_size: bool = True,
    include_chain: bool = True,
    include_repeat: bool = True,
    include_concat: bool = False,
) -> List[Augmenter]:
    basic_augmenters_to_apply = (
        [
            Rotate(90),
            Rotate(270),
            Rotate(180),
            Flip(0),
            Flip(1),
            Reflect(0, reverse=True),
            Reflect(1, reverse=True),
            Reflect(0, reverse=False),
            Reflect(1, reverse=False),
            RandomTranslateXY(),
            Transpose(),
        ]
        if include_basic
        else []
    )

    size_augmenters_to_apply = (
        [
            IncreaseResolution(2),
            IncreaseHeight(2),
            IncreaseWidth(2),
        ]
        if include_size
        else []
    )

    concat_augmenters_to_apply = (
        [
            Concat((IdentityAugmenter(), Rotate(180)), axis=0),
            Concat((IdentityAugmenter(), Rotate(180)), axis=1),
        ]
        if include_concat
        else []
    )

    chain_augmenters_to_apply = (
        [
            Chain([Rotate(90), IncreaseResolution(2)]),
            Chain([Rotate(270), IncreaseResolution(2)]),
            Chain([Rotate(180), IncreaseResolution(2)]),
            Chain([Flip(0), IncreaseResolution(2)]),
            Chain([Flip(1), IncreaseResolution(2)]),
            Chain([Transpose(), IncreaseResolution(2)]),
        ]
        if include_chain
        else []
    )

    repeat_augmenters_to_apply = (
        [
            Repeat(0, 2),
            Repeat(1, 2),
            Repeat(2, 2),
        ]
        if include_repeat
        else []
    )

    augmenters_to_apply = (
        basic_augmenters_to_apply
        + size_augmenters_to_apply
        + concat_augmenters_to_apply
        + chain_augmenters_to_apply
        + repeat_augmenters_to_apply
    )

    print(
        "Augmenters to apply: ", augmenters_to_apply, "len: ", len(augmenters_to_apply)
    )
    return augmenters_to_apply


def format_and_filter(formatter, task,):
    task = formatter.encode(task)
    data = {"input": task[0], "output": task[1]}
    return data


def get_test_time_train_data(
    original_task: Task, augmenters: List[Augmenter], n: int = 1, permute_n: int = 1, seed: int = 0
) -> List[Task]:
    rng = np.random.RandomState(seed)
    train_examples = original_task.train_examples.copy()
    initial_tasks = []
    N = len(train_examples)
    for i in range(len(train_examples)):
        examples = train_examples.copy()
        indices = set(range(N)) - {i}
        # we already remove i, so we need to remove n-1 more
        combs = list(itertools.combinations(indices, n - 1))
        combs = [indices - set(comb) for comb in combs]

        for comb in combs:
            initial_tasks.append(
                Task(name="", train_examples=[examples[j] for j in comb], test_example=examples[i])
            )

    augmented_tasks = []
    for augmenter in augmenters:
        for task in initial_tasks:
            task = augmenter.apply_to_task(task, to_input=True, to_output=True, rng=rng)
            # some augmentations increase shapes
            if not (task.max_height() <= 30 and task.max_width() <= 30):
                continue
            augmented_tasks.append(task)

    augmented_tasks = list(set(augmented_tasks + initial_tasks))

    color_and_permute_augmented_tasks = []

    for _ in range(permute_n):
        for task in augmented_tasks:
            if len(augmenters) != 0:
                new_task = PermuteColors().apply_to_task(task, to_input=True, to_output=True, rng=rng)
            else:
                new_task = task
            new_task = PermuteExamples().apply_to_task(
                new_task, rng=rng, to_input=True, to_output=True
            )
            color_and_permute_augmented_tasks.append(new_task)

    augmented_tasks = color_and_permute_augmented_tasks + augmented_tasks

    augmented_tasks = list(set(augmented_tasks))

    return augmented_tasks


def get_formatted_data(
    task: Task,
    augmenters: List[Augmenter],
    formatter: MessageRepresenter,
    leave_n: int = 1,
    permute_n: int = 1,
    seed: int = 0,
):

    train_data = get_test_time_train_data(
        task, augmenters, n=leave_n, permute_n=permute_n, seed=seed
    )

    formatted_data = []
    for task in train_data:
        formatted = format_and_filter(formatter, task,)
        formatted_data.append(formatted)

    return formatted_data


def process_task(
    task: Task,
    augmenters: List[Augmenter],
    formatter: MessageRepresenter,
    permute_n: int = 1,
    Nmax: int = 250,
    seed: int = 0,
):
    rng = np.random.RandomState(seed)

    leave_1_train_data = get_formatted_data(
        task, augmenters, formatter, leave_n=1, permute_n=permute_n, seed=seed
    )
    leave_2_train_data = get_formatted_data(
        task, augmenters, formatter, leave_n=2, permute_n=permute_n, seed=seed
    )

    train = leave_1_train_data

    if len(train) == 0:
        train = leave_2_train_data
    elif len(train) < Nmax:
        train += leave_2_train_data[: Nmax - len(train)]
    elif len(train) > Nmax:
        rng.shuffle(train)
        train = train[:Nmax]

    return train