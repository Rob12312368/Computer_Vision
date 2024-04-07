import numpy as np

# Create a 3x3 array
array = np.array([[5,6,1],
                  [8, 9, 1],
                  [1, 1, 1]])

# Compute the gradient along the x-axis (axis=1)
gradient_x = np.gradient(np.gradient(array, axis=1),axis=1)

# Compute the gradient along the y-axis (axis=0)
gradient_y = np.gradient(np.gradient(array, axis=0),axis=0)

print("Gradient along the x-axis:")
print(np.sum(gradient_x))

print("\nGradient along the y-axis:")
print(np.sum(gradient_y))
