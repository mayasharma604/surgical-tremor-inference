import numpy as np

def generate_clean_trajectory(T=500):
    t = np.linspace(0, 1, T)
    x = t
    y = 0.5 * np.sin(2 * np.pi * t)
    return np.stack([x, y], axis=1)

def add_tremor(traj, freq=25, amp=0.01):
    t = np.linspace(0, 1, len(traj))
    tremor = amp * np.sin(2 * np.pi * freq * t)
    tremor = np.stack([tremor, tremor], axis=1)
    return traj + tremor + np.random.normal(0, amp / 4, traj.shape)

def generate_dataset(n_samples=200, T=500):
    noisy, clean = [], []
    for _ in range(n_samples):
        base = generate_clean_trajectory(T)
        noisy.append(add_tremor(base))
        clean.append(base)
    return np.array(noisy), np.array(clean)

if __name__ == "__main__":
    noisy, clean = generate_dataset()
    np.save("data/noisy.npy", noisy)
    np.save("data/clean.npy", clean)
