from convert_to_p7_format import get_IMU_df, get_emg_df,convert_raw_values
import os
import pandas as pd
import numpy as np
import yaml
#load yaml 
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

saving_csv_name = f"converted_{config['sensor_type']}_{config['axis']}.csv" if config['sensor_type'] != "emg" else "converted_emg.csv"

def merge_subjects(type="emg"):
    dfs = []
    for subject in config['participants_list_ids']:
        subject_folder = os.path.join(config["converted_data_path"], subject)
        merged_df = pd.DataFrame()
        if type == "emg":
            csv_path = os.path.join(subject_folder, saving_csv_name)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                merged_df = pd.concat([merged_df, df], axis=1)
        else:
            for ax in config['axis']:
                csv_path = os.path.join(subject_folder, f"converted_{config['sensor_type']}_{ax}.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    merged_df = pd.concat([merged_df, df], axis=1)
        dfs.append(merged_df)

    df= pd.concat(dfs, axis=0)
    name_save = f'merged_{type}.csv'
    save_path = os.path.join(config["converted_data_path"], name_save)
    df.to_csv(save_path, index=True)
    print(len(df))
    print(df.head())
    print(f"Merged dataframe saved to {save_path}")

if __name__ == "__main__":
    for participant_id in config['participants_list_ids']:
        participant_folder = os.path.join(config["converted_data_path"], participant_id)
        # check if already converted file exists 
        if os.path.exists(os.path.join(participant_folder, saving_csv_name)):
            print(f"Converted file already exists for participant: {participant_id}. Skipping conversion.")
            continue
        if os.path.exists(participant_folder):
            pass
        else:
            os.makedirs(participant_folder, exist_ok=True)
        print(f"Processing participant: {participant_id}")
        csv_path1 = os.path.join(os.path.join(config["raw_data_path"], participant_id), config["df_raw_name"])
        print(f"Reading data from: {csv_path1}")
        df1 = pd.read_csv(csv_path1)
        df1 = convert_raw_values(df1, normalize=False)
        print(f"Converted raw values for participant: {participant_id}.")
        if config['sensor_type'] == "emg":
            emg_df = get_emg_df(df1,saving_dir=participant_folder)
        else:
            imu_df = get_IMU_df(df1, config['sensor_type'], config['axis'],saving_dir=participant_folder)
            print(f"Converted {config['sensor_type']} data along {config['axis']} for participant: {participant_id}.")
        print(f"Conversion completed for participant: {participant_id}.")

    merge_subjects(config['sensor_type'])

#todo add merge function per person, for all participants.
#todo adding df payin functionality