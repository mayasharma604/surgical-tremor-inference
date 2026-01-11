import matplotlib.pyplot as plt
import numpy as np
import time
import csv

OUTPUT_FILE = "data/tracing_task.csv"

# Generate target path (intent)
t = np.linspace(0, 1, 500)
target_x = t
target_y = 0.5 * np.sin(2 * np.pi * t)

# To record mouse positions
recorded = []
start_time = None

def on_move(event):
    global start_time
    if event.xdata is None or event.ydata is None:
        return
    if start_time is None:
        start_time = time.time()
    timestamp = time.time() - start_time
    recorded.append((timestamp, event.xdata, event.ydata))

# Set up figure
fig, ax = plt.subplots()
ax.plot(target_x, target_y, linestyle="--", label="Target Path")
ax.set_title("Trace the dashed line as accurately as possible")
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-1, 1)
ax.legend()

fig.canvas.mpl_connect("motion_notify_event", on_move)

plt.show()

# Save recorded mouse movement
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "x", "y"])
    writer.writerows(recorded)

print(f"Saved tracing data to {OUTPUT_FILE}")
