import requests
import time
from datetime import datetime
import sys
import pandas as pd

class GPUPowerRecorder:
    def __init__(self, duration, output_file, server_url="http://localhost:8085/data.json", sample_interval=0.1):
        """
        Initializes the GPUPowerRecorder.

        :param duration: Duration to record in seconds.
        :param output_file: Name of the output Excel file (without extension).
        :param server_url: URL to fetch JSON data from LibreHardwareMonitor.
        :param sample_interval: Time between samples in seconds.
        """
        self.duration = duration
        self.output_file = output_file if output_file.lower().endswith('.xlsx') else output_file + '.xlsx'
        self.server_url = server_url
        self.sample_interval = sample_interval
        self.data = []
        self.start_time = time.time()
        self.end_time = self.start_time + self.duration

    def fetch_json_data(self):
        """
        Fetches JSON data from the LibreHardwareMonitor web server.

        :return: Parsed JSON data or None if an error occurs.
        """
        try:
            response = requests.get(self.server_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"\nError fetching data from LibreHardwareMonitor: {e}")
            return None
        except ValueError as e:
            print(f"\nError parsing JSON data: {e}")
            return None

    def traverse_and_extract_power_sensors(self, node, hardware_path="", power_sensors=None):
        """
        Recursively traverses the JSON data to extract all power sensors.

        :param node: Current node in the JSON data.
        :param hardware_path: Accumulated path of hardware components.
        :param power_sensors: List to accumulate power sensor data.
        :return: List of tuples containing (Sensor Name, Power Value).
        """
        if power_sensors is None:
            power_sensors = []

        if isinstance(node, dict):
            text = node.get("Text", "")
            sensor_type = node.get("Type", "")
            value_str = node.get("Value", "")

            # Update hardware path when encountering a hardware component
            if "children" not in node and "Powers" not in text and node.get("Children"):
                new_hardware_path = f"{hardware_path} > {text}" if hardware_path else text
            else:
                new_hardware_path = hardware_path

            # Check if the node is a Power sensor
            if sensor_type == "Power" and text:
                # Generate a unique sensor name by combining hardware path and sensor name
                sensor_name = f"{hardware_path} > {text}" if hardware_path else text
                try:
                    # Extract numerical value (assumes format like "54.0 W")
                    power_value = float(value_str.split()[0])
                except (ValueError, IndexError):
                    power_value = 0.0  # Default to 0.0 if parsing fails

                power_sensors.append((sensor_name, power_value))

            # Recursively traverse child nodes
            for child in node.get("Children", []):
                self.traverse_and_extract_power_sensors(child, new_hardware_path, power_sensors)

        elif isinstance(node, list):
            for item in node:
                self.traverse_and_extract_power_sensors(item, hardware_path, power_sensors)

        return power_sensors

    def record(self):
        """
        Starts the recording process, fetching power data at regular intervals.
        """
        print(f"Recording power consumption for {self.duration} seconds...")
        print(f"Press Ctrl+C to stop recording early.")
        try:
            while time.time() < self.end_time:
                json_data = self.fetch_json_data()
                if json_data:
                    power_sensors = self.traverse_and_extract_power_sensors(json_data)
                    if power_sensors:
                        current_time = time.time()
                        timestamp = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S")
                        elapsed_time = current_time - self.start_time
                        record = {"Timestamp": timestamp, "Elapsed Time (s)": round(elapsed_time, 3)}
                        for sensor_name, power in power_sensors:
                            # Sanitize sensor name for Excel (remove characters like '>' and replace with '_')
                            sanitized_name = sensor_name.replace(" > ", " - ").replace("/", "_").replace("{", "").replace("}", "")
                            record[sanitized_name] = power
                        self.data.append(record)

                        # Print current power readings
                        print(f"\rElapsed Time: {elapsed_time:.3f}s | ", end="")
                        for sensor_name, power in power_sensors:
                            print(f"{sensor_name}: {power:.2f}W | ", end="")
                        sys.stdout.flush()
                    else:
                        print("\rNo power sensor data available. ", end="")
                else:
                    print("\rFailed to fetch data. ", end="")
                time.sleep(self.sample_interval)
        except KeyboardInterrupt:
            print("\nRecording stopped by user.")

        if self.data:
            # Create a DataFrame
            df = pd.DataFrame(self.data)

            # Reorder columns: Timestamp, Elapsed Time, then sensors
            cols = ['Timestamp', 'Elapsed Time (s)'] + [col for col in df.columns if col not in ['Timestamp', 'Elapsed Time (s)']]
            df = df[cols]

            # Save to Excel
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
