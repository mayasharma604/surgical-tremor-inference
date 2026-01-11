import numpy as np
import torch
from model.intent_lstm import IntentLSTM
from kalman import KalmanFilter2D

# Load data
noisy = np.load("data/noisy_mouse_data.npy")
clean = np.load("data/clean_mouse_data.npy")

# Load ML model
model = IntentLSTM()
model.load_state_dict(torch.load("intent_model.pt"))
model.eval()

# ML predictions
window = 20
ml_preds = []
for t in range(len(noisy)-window):
    x_input = torch.tensor(noisy[t:t+window]).unsqueeze(0).float()
    pred = model(x_input).detach().numpy()[0]
    ml_preds.append(pred)
ml_preds = np.array(ml_preds)

# Kalman Filter predictions
kf_x = KalmanFilter2D()
kf_y = KalmanFilter2D()
kf_preds = []
for xy in noisy:
    pred_x = kf_x.step(xy[0])
    pred_y = kf_y.step(xy[1])
    kf_preds.append([pred_x[0], pred_y[0]])
kf_preds = np.array(kf_preds)

# Compute MSE
def mse(a, b):
    return np.mean((a - b)**2)

print("ML MSE:", mse(ml_preds, clean[window:]))
print("Kalman MSE:", mse(kf_preds, clean))
