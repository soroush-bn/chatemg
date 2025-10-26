from convert_to_p7_format import get_IMU_df, get_emg_df,convert_raw_values
import os
import pandas as pd
import numpy as np

participants_list_ids = ["033106b27b","bc4dd952fe","31afab1e30","97c6aaac2d","7037a93026","98aa5fac2d","ecfa481b42","e49db6578f","27f6898a3f","3f858df9cf","9780ed81f4"] #"bc4dd952fe","31afab1e30","97c6aaac2d","7037a93026","98aa5fac2d"]
data_path = "/home/sbaghernezha/data/"
csv_name = "final_df.csv"
csv_name_down= "final_df_payin.csv"
sensor_type = "emg"
axis = ["x"]  
saving_dir =  "../data/"


def merge_subjects(type="emg"):
    dfs = []
    for subject in participants_list_ids:
        subject_folder = os.path.join(saving_dir, subject)
        merged_df = pd.DataFrame()
        if type == "emg":
            csv_file = 'converted_emg.csv'
            csv_path = os.path.join(subject_folder, csv_file)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                merged_df = pd.concat([merged_df, df], axis=1)
        else:
            for ax in axis:
                csv_file = f'converted_{sensor_type}_{ax}.csv'
                csv_path = os.path.join(subject_folder, csv_file)
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    merged_df = pd.concat([merged_df, df], axis=1)
        dfs.append(merged_df)

    df= pd.concat(dfs, axis=0)
    name_save = f'merged_{type}.csv'
    save_path = os.path.join(saving_dir, name_save)
    df.to_csv(save_path, index=True)
    print(len(df))
    print(df.head())
    print(f"Merged dataframe saved to {save_path}")

if __name__ == "__main__":
    for participant_id in participants_list_ids:
        participant_folder = os.path.join(saving_dir, participant_id)
        #check if already converted file exists 
        if os.path.exists(os.path.join(participant_folder, csv_name)):
            print(f"Converted file already exists for participant: {participant_id}. Skipping conversion.")
            continue
        if os.path.exists(participant_folder):
            pass
        else:
            os.makedirs(participant_folder, exist_ok=True)
        print(f"Processing participant: {participant_id}")
        csv_path1 = os.path.join(os.path.join(data_path, participant_id),csv_name)
        print(f"Reading data from: {csv_path1}")
        df1 = pd.read_csv(csv_path1)
        df1 = convert_raw_values(df1, normalize=False)
        print(f"Converted raw values for participant: {participant_id}.")
        if sensor_type == "emg":
            emg_df = get_emg_df(df1,saving_dir=participant_folder)
        else:
            for ax in axis:
                imu_df = get_IMU_df(df1, sensor_type, ax,saving_dir=participant_folder)
                print(f"Converted {sensor_type} data along {ax}-axis for participant: {participant_id}.")
        print(f"Conversion completed for participant: {participant_id}.")
    
    merge_subjects("accel")

#todo add merge function per person, for all participants.
#todo adding df payin functionality