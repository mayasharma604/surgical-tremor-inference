import numpy as np
import pandas as pd
from tremor_profiles import generate_physiological_tremor

INPUT_FILE = "data/raw_mouse_data.csv"
OUTPUT_FILE = "data/noisy_mouse_data.npy"

df = pd.read_csv(INPUT_FILE)

t = df["time"].values
x = df["x"].values
y = df["y"].values

tremor_x, tremor_y = generate_physiological_tremor(t)

noisy_x = x + tremor_x
noisy_y = y + tremor_y

trajectory = np.stack([noisy_x, noisy_y], axis=1)
np.save(OUTPUT_FILE, trajectory)

print(f"Saved tremor-injected trajectory to {OUTPUT_FILE}")
