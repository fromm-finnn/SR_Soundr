import numpy as np
import os

file_path = "data/output.npy"
file_size = os.path.getsize(file_path)
print(f"File size: {file_size / (1024 * 1024):.2f} MB")

try:
    data = np.load("data/output.npy", mmap_mode='r')
    print("Shape (mmap):", data.shape)
except Exception as e:
    print("Error with mmap_mode:", e)