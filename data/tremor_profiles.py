import numpy as np

def generate_physiological_tremor(t, freq_range=(8, 12), amp_range=(0.3, 1.0)):
    freq = np.random.uniform(*freq_range)
    amp = np.random.uniform(*amp_range)

    tremor = amp * np.sin(2 * np.pi * freq * t)

    # correlated x/y tremor (more realistic)
    tremor_x = tremor
    tremor_y = tremor * np.random.uniform(0.7, 1.0)

    return tremor_x, tremor_y
