import numpy as np

import numpy as np

def generate_laplace_kernel(size):
    """
    Generate a Laplace kernel of a given size.
    
    Args:
        size (int): The size of the kernel (odd number).
    
    Returns:
        numpy.ndarray: The Laplace kernel.
    """
    # Check if the size is an odd number
    if size % 2 == 0:
        raise ValueError("The size of the kernel must be an odd number.")
    
    # Initialize the kernel with zeros
    kernel = np.zeros((size, size), dtype=int)
    
    # Calculate the center position
    center = size // 2
    
    # Set the values of the kernel
    kernel[center, center] = -4
    for i in range(size):
        kernel[center, i] += 1
        kernel[i, center] += 1
    
    return kernel
print(generate_laplace_kernel(3))
