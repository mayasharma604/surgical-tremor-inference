import numpy as np

noisy = np.load("data/noisy_mouse_data.npy")
clean = np.load("data/clean_mouse_data.npy")

print("Noisy shape:", noisy.shape)
print("Clean shape:", clean.shape)