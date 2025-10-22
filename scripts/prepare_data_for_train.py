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
    for participant_id in participants_list_ids:
        participant_folder = os.path.join(saving_dir, participant_id)
        if os.path.exists(participant_folder):
            continue
        else:
            os.makedirs(participant_folder, exist_ok=True)
        print(f"Processing participant: {participant_id}")
        csv_path1 = os.path.join(os.path.join(data_path, participant_id),csv_name)
        print(f"Reading data from: {csv_path1}")
        df1 = pd.read_csv(csv_path1)
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


#todo add merge function per person, for all participants.
#todo adding df payin functionality