from convert_to_p7_format import get_IMU_df, get_emg_df,convert_raw_values
import os
import pandas as pd
import numpy as np

participants_list_ids = ["033106b27b","bc4dd952fe","31afab1e30","97c6aaac2d","7037a93026","98aa5fac2d","ecfa481b42","e49db6578f","27f6898a3f","3f858df9cf","9780ed81f4"] #"bc4dd952fe","31afab1e30","97c6aaac2d","7037a93026","98aa5fac2d"]
data_path = "/home/sbaghernezha/data/"
csv_name = "finl_df.csv"
csv_name_down= "finl_df_payin.csv"
sensor_types = ["accel"] 
axis = ["x"]  
saving_dir =  "../data/"
if __name__ == "__main__":
    import sys
    for participant_id in participants_list_ids:
        try:
            participant_folder = os.path.join(saving_dir, participant_id)
            if os.path.exists(participant_folder):
                print(f"Skipping {participant_id} - folder already exists")
                continue
            else:
                os.makedirs(participant_folder, exist_ok=True)
            print(f"Processing participant: {participant_id}")
            csv_path1 = os.path.join(os.path.join(data_path, participant_id),csv_name)
            print(f"Reading data from: {csv_path1}")
            
            # Check file size before reading
            file_size = os.path.getsize(csv_path1) / (1024 * 1024)  # Size in MB
            print(f"File size: {file_size:.2f} MB")
            
            df1 = pd.read_csv(csv_path1)
            print(f"Data loaded successfully. Shape: {df1.shape}")
            print(f"Memory usage: {df1.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
            
            df1 = convert_raw_values(df1, normalize=False)
            print(f"Converted raw values for participant: {participant_id}.")
            if sensor_types == ["emg"]:
                emg_df = get_emg_df(df1,saving_dir=participant_folder)
            else:
                for sensor_type in sensor_types:
                    for ax in axis:
                        imu_df = get_IMU_df(df1, sensor_type, ax,saving_dir=participant_folder)
                        print(f"Converted {sensor_type} data along {ax}-axis for participant: {participant_id}.")
            print(f"Conversion completed for participant: {participant_id}.")
            
        except Exception as e:
            print(f"ERROR processing {participant_id}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue


#todo add merge function per person, for all participants.
#todo adding df payin functionality