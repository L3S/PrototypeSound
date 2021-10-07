import os
import pandas as pd
from scipy.signal import butter, lfilter
from utilities import read_audio, create_folder, write_audio
import config

# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def read_train_test_meta(meta_csv):
    df = pd.read_csv(meta_csv, sep='\t', header=None)
    df = pd.DataFrame(df)
    
    audio_names = []
    set_catogeries = []

    for row in df.iterrows():
        audio_name = row[1][0]
        set_category = row[1][1]
        
        audio_names.append(audio_name)
        set_catogeries.append(set_category)

    return audio_names, set_catogeries


def read_audio_meta(meta_csv):
    df = pd.read_csv(meta_csv, sep='\t', header=None)
    df = pd.DataFrame(df)

    begins = []
    ends = []
    crackles = []
    wheezes = []

    for row in df.iterrows():
        begin_cycle = row[1][0]
        end_cycle = row[1][1]
        crackle = row[1][2]
        wheeze = row[1][3]

        begins.append(begin_cycle)
        ends.append(end_cycle)
        crackles.append(crackle)
        wheezes.append(wheeze)

    return begins, ends, crackles, wheezes

def segmentation(data_dir, meta_dir, output_audio_dir, output_meta_dir):
    """
    Segment respiratory cycles and generate meta files.
    Args:
        data_dir: original audio files folder
        meta_dir: original meta files folder
        output_audio_dir: output audio folder
        output_meta_dir: output meta data folder

    Returns:

    """
    sample_rate = config.sample_rate
    bandpass_low = config.bandpass_low
    bandpass_high = config.bandpass_high

    audio_names_new = []
    set_categories_new = []
    cycle_labels = []

    audio_names, set_categories = read_train_test_meta(os.path.join(meta_dir, 'ICBHI_challenge_train_test.txt'))

    for audio_i in range(0, len(audio_names)):
        audio_name = audio_names[audio_i]
        set_category = set_categories[audio_i]

        if audio_name == '226_1b1_Pl_sc_Meditron':
            audio, fs = read_audio(os.path.join(data_dir, '226_1b1_Pl_sc_LittC2SE.wav'), target_fs=sample_rate)
            begins, ends, crackles, wheezes = read_audio_meta(os.path.join(data_dir, '226_1b1_Pl_sc_LittC2SE.txt'))
        else:
            audio, fs = read_audio(os.path.join(data_dir, audio_name+'.wav'), target_fs=sample_rate)
            begins, ends, crackles, wheezes = read_audio_meta(os.path.join(data_dir, audio_name+'.txt'))
        # butter bandpass filter
        audio = butter_bandpass_filter(audio, bandpass_low, bandpass_high, sample_rate)
        for cycle_i in range(0, len(begins)):
            begin_frame = int(begins[cycle_i] * sample_rate)
            end_frame = int(ends[cycle_i] * sample_rate)

            audio_seg = audio[begin_frame:end_frame]
            write_audio(os.path.join(output_audio_dir, audio_name + '_' + str(cycle_i) + '.wav'), audio_seg, sample_rate)

            audio_names_new.append(audio_name + '_' + str(cycle_i) + '.wav')
            set_categories_new.append(set_category)
            if crackles[cycle_i] == 0 and wheezes[cycle_i] == 0:
                cycle_labels.append('normal')
            elif crackles[cycle_i] == 0 and wheezes[cycle_i] == 1:
                cycle_labels.append('wheeze')
            elif crackles[cycle_i] == 1 and wheezes[cycle_i] == 0:
                cycle_labels.append('crackle')
            elif crackles[cycle_i] == 1 and wheezes[cycle_i] == 1:
                cycle_labels.append('both')
            else:
                print('Wrong labels!')

    df = pd.DataFrame(data={'audio_name': audio_names_new, 'set_category': set_categories_new, 'cycle_label': cycle_labels})
    df.to_csv(os.path.join(output_meta_dir, 'meta.csv'), index=False)

if __name__ == '__main__':
    data_dir = '../ICBHI_final_database/'
    meta_dir = '../ICBHI_meta/'

    output_dir = '../data_experiment/'
    output_audio_dir = os.path.join(output_dir, 'audio')
    output_meta_dir = os.path.join(output_dir, 'meta_data')
    create_folder(output_audio_dir)
    create_folder(output_meta_dir)

    segmentation(data_dir, meta_dir, output_audio_dir, output_meta_dir)
