import pandas as pd
import datetime
import numpy as np

label_mapping = {
    "Thumb Extension":0,"index Extension":1,"Middle Extension":2,"Ring Extension":3,
             "Pinky Extension":4,"Thumbs Up":5,"Right Angle":6,"Peace":7,"OK":8,"Horn":9,"Hang Loose":10,
             "Power Grip":11,"Hand Open":12,"Wrist Extension":13,"Wrist Flexion":14,"Ulnar deviation":15,"Radial Deviation":16    
}

df = pd.read_csv('final_df.csv')

new_df = pd.DataFrame()

for i in range(1,9):
    new_df[f'emg{i}'] = df[f'emg{i}']

new_df['gt'] = df['label'].map(label_mapping)  

current_time = datetime.datetime.now()
base_timestamp = current_time.strftime('%Y%m%d%H%M%S%f')[:17] 

num_rows = len(df)
total_time = 2.0  
time_per_row = total_time / num_rows
time_elapsed = np.arange(0, total_time, time_per_row)[:num_rows]

new_df['time_elapsed'] = time_elapsed
new_df['current_time'] = [int(base_timestamp) + i for i in range(len(df))]
new_df['current_task'] = 'None'

new_df = new_df[['gt', 'time_elapsed', 'current_time', 'current_task', 
                 'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8']]

new_df.to_csv('converted_final_df.csv', index=True)
print("Conversion completed. File saved as 'converted_final_df.csv'")