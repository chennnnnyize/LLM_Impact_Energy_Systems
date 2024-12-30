import subprocess
import time
from datetime import datetime
import sys
import pandas as pd

class CPUPowerRecorder:
    def __init__(self, duration, output_file):
        self.duration = duration
        self.output_file = output_file if output_file.lower().endswith('.xlsx') else output_file + '.xlsx'
        self.interval = 1  # Set your desired interval in seconds

    def record(self):
        print(f"Recording CPU power consumption for {self.duration} seconds...")
        print("Press Ctrl+C to stop recording early.")
        data = []
        start_time = time.perf_counter()  # High-resolution start time
        last_record_time = start_time

        try:
            # Start pcm-power process
            with subprocess.Popen(['sudo', 'pcm-power', '--csv'], stdout=subprocess.PIPE, text=True) as proc:
                while time.perf_counter() - start_time < self.duration:
                    current_time = time.perf_counter()
                    if current_time - last_record_time >= self.interval:
                        line = proc.stdout.readline().strip()
                        print(f"Debug Line Output: {line}")  # Debug line for inspection

                        if "Consumed energy units" in line:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            elapsed_time = current_time - start_time

                            try:
                                parts = line.split(';')
                                cpu_energy = float(parts[1].split(':')[1].strip())
                                dram_energy = float(parts[3].split(':')[1].strip())
                                print(f"Parsed Data - Timestamp: {timestamp}, Elapsed Time: {elapsed_time:.2f}s, CPU Energy: {cpu_energy} J, DIMM Energy: {dram_energy} J")
                            except (IndexError, ValueError) as e:
                                print(f"Warning: Could not parse energy data. Error: {e}")
                                continue

                            data.append({
                                "Timestamp": timestamp,
                                "Elapsed Time (s)": elapsed_time,
                                "CPU Energy (J)": cpu_energy,
                                "DIMM Energy (J)": dram_energy
                            })

                            print(f"\rElapsed Time: {elapsed_time:.3f}s | CPU Energy: {cpu_energy:.2f}J | DIMM Energy: {dram_energy:.2f}J", end="")
                            sys.stdout.flush()
                            
                            last_record_time = current_time  # Reset the last record time

        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
        
        if data:
            print("\nSample of Recorded Data:")
            for record in data:
                print(record)
            df = pd.DataFrame(data)
            df.to_excel(self.output_file, index=False)
            print(f"\nData saved to {self.output_file}")
        else:
            print("\nNo data recorded. Excel file was not created.")

if __name__ == "__main__":
    try:
        duration = float(input("Enter the duration to record (in seconds): "))
        output_file = input("Enter the output Excel file name (without extension): ")
        recorder = CPUPowerRecorder(duration, output_file)
        recorder.record()
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

