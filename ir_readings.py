import max30102
import time

m = max30102.MAX30102()
ir_data = []

# Collect data for 20 seconds at 50 Hz
for _ in range(20 * 50):
    red, ir = m.read_sequential()
    ir_data.append(ir[-1])  # Append the latest IR value
    time.sleep(0.02)  # 20 ms delay for 50 Hz sampling rate

# Save data to a text file
with open("ppg_data.txt", "w") as f:
    f.write(",".join(map(str, ir_data)))
