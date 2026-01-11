import time
import csv
from pynput import mouse

OUTPUT_FILE = "data/raw_mouse_data.csv"
SAMPLE_RATE_HZ = 120  # realistic teleop frequency

positions = []
start_time = None

def on_move(x, y):
    global start_time
    if start_time is None:
        start_time = time.time()
    timestamp = time.time() - start_time
    positions.append((timestamp, x, y))

listener = mouse.Listener(on_move=on_move)
listener.start()

print("Recording mouse data...")
print("Instructions:")
print("- Trace a shape or hold steady")
print("- Press Ctrl+C to stop recording")

try:
    while True:
        time.sleep(1 / SAMPLE_RATE_HZ)
except KeyboardInterrupt:
    listener.stop()
    print("Recording stopped.")

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "x", "y"])
    writer.writerows(positions)

print(f"Saved {len(positions)} samples to {OUTPUT_FILE}")
