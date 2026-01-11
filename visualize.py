import matplotlib.pyplot as plt
import numpy as np
from kalman import KalmanFilter2D

noisy = np.load("data/noisy_mouse_data.npy")
clean = np.load("data/clean_mouse_data.npy")

# Kalman baseline
kf_x = KalmanFilter2D()
kf_y = KalmanFilter2D()
kf_preds = []
for xy in noisy:
    pred_x = kf_x.step(xy[0])
    pred_y = kf_y.step(xy[1])
    kf_preds.append([pred_x[0], pred_y[0]])
kf_preds = np.array(kf_preds)

plt.figure(figsize=(6,6))
plt.plot(noisy[:,0], noisy[:,1], label="Noisy Input", alpha=0.5)
plt.plot(clean[:,0], clean[:,1], label="Ground Truth Intent")
plt.plot(kf_preds[:,0], kf_preds[:,1], label="Kalman Baseline", linestyle="--")
plt.legend()
plt.title("ML vs Kalman Tremor Suppression Baseline")
plt.show()
