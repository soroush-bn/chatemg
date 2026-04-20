import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import consts


class Visualizer:
    def __init__(self):
        self.emg_channels = ['emg0', 'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7']
        self.colors = plt.cm.tab10(np.linspace(0, 1, 8))

    def load_csv(self, filepath):
        """Load CSV file and return as DataFrame"""
        return pd.read_csv(filepath)

    def plot_synthetic_gestures(self, synthetic_data, samples_per_rep=None, num_reps=None, subject=1, title="Synthetic EMG Gestures"):

        if isinstance(synthetic_data, str):
            df = self.load_csv(synthetic_data)
        else:
            df = synthetic_data
        
        if num_reps is None:
            num_reps = consts.NUMBER_OF_REPEATS
        
        total_samples = len(df)
        samples_per_subject = total_samples // consts.NUMBER_OF_SUBJECTS
        
        start_idx = (subject - 1) * samples_per_subject
        end_idx = subject * samples_per_subject
        df_subject = df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # Determine channels to plot
        channels = [col for col in df_subject.columns if col.startswith('emg') or col.isdigit()]
        
        if not channels:
            raise ValueError("No EMG channels found in data")
        
        # Auto-calculate samples per rep if not provided
        if samples_per_rep is None:
            samples_per_rep = len(df_subject) // num_reps
        
        # Create subplots for each channel
        fig, axes = plt.subplots(len(channels), 1, figsize=(14, 2*len(channels)), sharex=True)
        
        if len(channels) == 1:
            axes = [axes]
        
        # Plot each channel
        for idx, channel in enumerate(channels):
            axes[idx].plot(df_subject[channel], color=self.colors[idx % len(self.colors)], linewidth=0.8)
            axes[idx].set_ylabel(f'{channel}', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            
            # Add vertical lines to separate reps
            for rep in range(1, num_reps):
                axes[idx].axvline(x=rep * samples_per_rep, color='red', linestyle='--', alpha=0.5)
        
        axes[-1].set_xlabel('Time (samples)', fontsize=11)
        axes[0].set_title(f"{title} - Subject {subject}", fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        return fig



    def save_plot(self, fig, filepath, dpi=300):
        """Save figure to file"""
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {filepath}")


if __name__== "__main__":
    # Example usage
    viz = Visualizer()
    data_path = r"E:\projects\chatemgserver\chatemg\data\synthetic_data_0_both_100k.csv"
    df = viz.load_csv(data_path)
    
    fig1 = viz.plot_synthetic_gestures(
        df, 
        subject=6,
        title="Synthetic EMG Data - Forearm (ds2_vs1000_bs512)"
    )

    # #do this for ds 2 as well
    # data_path2 = r".csv"
    # df2 = viz.load_csv(data_path2)
    # fig2 = viz.plot_synthetic_gestures(
    #     df2, 
    #     subject=1,
    #     title="Synthetic EMG Data - Forearm (ds4_vs1000_bs512)"
    # )

    # #do this for ds 10 as well 
    # data_path3 = r"E:\projects\chatemgserver\chatemg\data\synthetic_data_lastrep_0_forearm_ds10_vs1000_bs512.csv"
    # df3 = viz.load_csv(data_path3)
    # fig3 = viz.plot_synthetic_gestures(
    #     df3, 
    #     subject=1,
    #     title="Synthetic EMG Data - Forearm (ds10_vs1000_bs512)"
    # )

    # data_path4 = r"E:\projects\chatemgserver\chatemg\data\synthetic_data_0_forearm_ds10_vs1000_bs512.csv"
    # df4 = viz.load_csv(data_path4)
    # fig4 = viz.plot_synthetic_gestures(
    #     df4, 
    #     subject=1,
    #     title="Synthetic EMG Data - 10 percent Forearm (ds10_vs1000_bs512)"
    # )
    
  
    
    plt.show()