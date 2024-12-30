import subprocess
import time
from datetime import datetime
import sys
import pandas as pd

def get_gpu_power():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        return [float(power.strip()) for power in result.stdout.split('\n') if power.strip()]
    except subprocess.CalledProcessError:
        print("Error: Unable to get GPU power. Make sure nvidia-smi is installed and working.")
        return []

def get_gpu_count():
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True, check=True)
        return len(result.stdout.strip().split('\n'))
    except subprocess.CalledProcessError:
        print("Error: Unable to get GPU count. Make sure nvidia-smi is installed and working.")
        return 0

class GPUPowerRecorder:
    def __init__(self, duration, output_file):
        self.duration = duration
        self.output_file = output_file if output_file.lower().endswith('.xlsx') else output_file + '.xlsx'
        self.gpu_count = get_gpu_count()
        
        if self.gpu_count == 0:
            raise ValueError("No GPUs detected. Unable to record power consumption.")

    def record(self):
        print(f"Recording GPU power consumption for {self.duration} seconds...")
        print(f"Press Ctrl+C to stop recording early.")
        data = []
        start_time = time.time()
        next_sample_time = start_time

        try:
            while time.time() - start_time < self.duration:
                current_time = time.time()
                if current_time >= next_sample_time:
                    powers = get_gpu_power()
                    if powers:
                        timestamp = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S")
                        elapsed_time = current_time - start_time
                        record = {"Timestamp": timestamp, "Elapsed Time (s)": elapsed_time}
                        for i, power in enumerate(powers):
                            record[f"GPU {i} Power (W)"] = power
                        data.append(record)

                        # Print current power readings
                        print(f"\rElapsed Time: {elapsed_time:.3f}s | ", end="")
                        for i, power in enumerate(powers):
                            print(f"GPU {i}: {power:.2f}W | ", end="")
                        sys.stdout.flush()

                    next_sample_time += 0.1  # Set next sample time to exactly 0.1 second later

        except KeyboardInterrupt:
            print("\nRecording stopped by user.")

        if data:
            df = pd.DataFrame(data)
            df.to_excel(self.output_file, index=False)
            print(f"\nData saved to {self.output_file}")
        else:
            print("\nNo data was recorded. Excel file was not created.")

if __name__ == "__main__":
    try:
        duration = float(input("Enter the duration to record (in seconds): "))
        output_file = input("Enter the output Excel file name (without extension): ")
        recorder = GPUPowerRecorder(duration, output_file)
        recorder.record()
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")