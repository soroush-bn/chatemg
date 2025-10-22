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
    import traceback as tb
    
    print("="*70)
    print("STARTING DATA PREPARATION")
    print("="*70)
    print(f"Total participants to process: {len(participants_list_ids)}")
    print(f"Data path: {data_path}")
    print(f"Saving directory: {saving_dir}")
    print(f"Sensor types: {sensor_types}")
    print(f"Axis: {axis}")
    print("="*70)
    sys.stdout.flush()
    
    successful = []
    failed = []
    skipped = []
    
    for idx, participant_id in enumerate(participants_list_ids, 1):
        print(f"\n[{idx}/{len(participants_list_ids)}] Processing: {participant_id}")
        sys.stdout.flush()
        
        try:
            participant_folder = os.path.join(saving_dir, participant_id)
            print(f"  → Checking folder: {participant_folder}")
            sys.stdout.flush()
            
            if os.path.exists(participant_folder):
                print(f"  ⊘ SKIPPED - folder already exists")
                skipped.append(participant_id)
                sys.stdout.flush()
                continue
            else:
                os.makedirs(participant_folder, exist_ok=True)
                print(f"  ✓ Created folder")
                sys.stdout.flush()
            
            csv_path1 = os.path.join(data_path, participant_id, csv_name)
            print(f"  → CSV path: {csv_path1}")
            sys.stdout.flush()
            
            # Check if file exists
            if not os.path.exists(csv_path1):
                raise FileNotFoundError(f"CSV file not found at: {csv_path1}")
            print(f"  ✓ File exists")
            sys.stdout.flush()
            
            # Check file size
            file_size = os.path.getsize(csv_path1) / (1024 * 1024)
            print(f"  → File size: {file_size:.2f} MB")
            sys.stdout.flush()
            
            # Read CSV
            print(f"  → Reading CSV...")
            sys.stdout.flush()
            df1 = pd.read_csv(csv_path1)
            print(f"  ✓ CSV loaded - Shape: {df1.shape}")
            sys.stdout.flush()
            
            # Check memory usage
            mem_usage = df1.memory_usage(deep=True).sum() / (1024*1024)
            print(f"  → Memory usage: {mem_usage:.2f} MB")
            sys.stdout.flush()
            
            # Convert raw values
            print(f"  → Converting raw values...")
            sys.stdout.flush()
            df1 = convert_raw_values(df1, normalize=False)
            print(f"  ✓ Raw values converted")
            sys.stdout.flush()
            
            # Process data
            if sensor_types == ["emg"]:
                print(f"  → Processing EMG data...")
                sys.stdout.flush()
                emg_df = get_emg_df(df1, saving_dir=participant_folder)
                print(f"  ✓ EMG data saved")
                sys.stdout.flush()
            else:
                for sensor_type in sensor_types:
                    for ax in axis:
                        print(f"  → Processing {sensor_type}_{ax}...")
                        sys.stdout.flush()
                        imu_df = get_IMU_df(df1, sensor_type, ax, saving_dir=participant_folder)
                        print(f"  ✓ {sensor_type}_{ax} saved")
                        sys.stdout.flush()
            
            print(f"  ✓✓✓ SUCCESS: {participant_id}")
            successful.append(participant_id)
            sys.stdout.flush()
            
        except Exception as e:
            error_msg = f"  ✗✗✗ ERROR: {str(e)}"
            print(error_msg)
            print(error_msg, file=sys.stderr)
            tb.print_exc()
            tb.print_exc(file=sys.stderr)
            failed.append((participant_id, str(e)))
            sys.stdout.flush()
            sys.stderr.flush()
            continue
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total participants: {len(participants_list_ids)}")
    print(f"Successful: {len(successful)}")
    print(f"Skipped (already processed): {len(skipped)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✓ Successfully processed ({len(successful)}):")
        for p in successful:
            print(f"  • {p}")
    
    if skipped:
        print(f"\n⊘ Skipped ({len(skipped)}):")
        for p in skipped:
            print(f"  • {p}")
    
    if failed:
        print(f"\n✗ Failed ({len(failed)}):")
        for p, err in failed:
            print(f"  • {p}: {err}")
    
    print("="*70)
    sys.stdout.flush()


#todo add merge function per person, for all participants.
#todo adding df payin functionality