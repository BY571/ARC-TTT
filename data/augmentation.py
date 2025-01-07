import random
from typing import List, Set

def augment_matrix(matrix: List[List[int]], num_augmentations: int = 1, max_number: int = 9) -> List[List[List[int]]]:
    """
    Create augmented versions of the input matrix by replacing unique non-zero integers with different numbers.

    This function generates a specified number of augmented matrices from the input matrix.
    Each augmentation replaces the unique non-zero integers in the original matrix with
    different integers, ensuring that the original numbers are not used in the augmentations.

    Parameters:
    -----------
    matrix : List[List[int]]
        The input matrix to be augmented. Can be of any size, including non-square matrices.
    num_augmentations : int, optional
        The number of augmented matrices to generate (default is 1).
    max_number : int, optional
        The maximum integer value to use for augmentations e.g. [1, ..., max_number] (default is 9).

    Returns:
    --------
    List[List[List[int]]]
        A list of augmented matrices. Each augmented matrix has the same dimensions as the input matrix.

    Raises:
    -------
    ValueError:
        If max_number is not a positive integer.
        If the input matrix contains values greater than max_number.
        If num_augmentations is not a positive integer.
        If num_augmentations exceeds the maximum possible unique augmentations.

    Example:
    --------
    >>> matrix = [[0, 7, 7], [7, 7, 7], [0, 7, 7]]
    >>> augmented = augment_matrix(matrix, num_augmentations=2, max_number=9)
    >>> print(augmented)
    [[[0, 1, 1], [1, 1, 1], [0, 1, 1]], [[0, 2, 2], [2, 2, 2], [0, 2, 2]]]
    """

    # Validate max_number
    if not isinstance(max_number, int) or max_number < 1:
        raise ValueError("max_number must be a positive integer.")

    # Find unique non-zero values in the matrix
    unique_values: Set[int] = set()
    for row in matrix:
        unique_values.update(set(row) - {0})
    
    # Ensure all values in the matrix are within the valid range
    if any(val > max_number for val in unique_values):
        raise ValueError(f"Matrix contains values greater than the specified max_number ({max_number}).")

    # Calculate the available numbers for augmentation
    available_numbers = list(set(range(1, max_number + 1)) - unique_values)
    
    # Calculate the maximum number of possible augmentations
    max_augmentations = min(len(available_numbers) ** len(unique_values), 1000000)
    
    # Validate the number of augmentations
    if not isinstance(num_augmentations, int) or num_augmentations <= 0:
        raise ValueError("Number of augmentations must be a positive integer.")
    if num_augmentations > max_augmentations:
        raise ValueError(f"Number of augmentations cannot exceed {max_augmentations} for this matrix.")
    
    # Create a list to store augmentations
    augmentations = []
    
    # Keep track of used augmentations to avoid duplicates
    used_augmentations = set()
    
    for _ in range(num_augmentations):
        # Create a mapping of unique values to new values
        while True:
            replacement_values = tuple(random.sample(available_numbers, len(unique_values)))
            if replacement_values not in used_augmentations:
                used_augmentations.add(replacement_values)
                break
        
        value_map = dict(zip(unique_values, replacement_values))
        
        # Create the augmented matrix
        augmented_matrix = [
            [value_map.get(cell, cell) for cell in row]
            for row in matrix
        ]
        
        augmentations.append(augmented_matrix)
    
    return augmentations


# Example usage
input_matrix = [[0, 7, 7], [7, 7, 7], [0, 7, 7]]
result = augment_matrix(input_matrix, num_augmentations=9, max_number=10)
print("Original matrix:", input_matrix)
print("Augmentations:", result)

# Another example 4x4 matrix
input_matrix2 = [[0, 7, 7], [1, 1, 7], [0, 7, 7]]
result2 = augment_matrix(input_matrix2, num_augmentations=2)
print("\nOriginal matrix:", input_matrix2)
print("Augmentations:", result2)

# Another example 4x3 matrix
input_matrix2 = [[0, 7, 7], [1, 1, 7], [0, 7, 7], [0, 0, 7]]
result2 = augment_matrix(input_matrix2, num_augmentations=2)
print("\nOriginal matrix:", input_matrix2)
print("Augmentations:", result2)