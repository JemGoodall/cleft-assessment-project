'''https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/'''
# ----------------------------
# Transforms directory of wav files into spectrograms
# resamples and pads the files if necessary
# and splits them into train and test directories if pass --split

# run as
# python make_spec_data [INPUT_WAV_DIR] [OUTPUT_SPEC DIR/folder name] --split
# ----------------------------

import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
from torch.utils.data import DataLoader, Dataset, random_split
import os
import sys
from pathlib import Path
import argparse
import numpy as np
import librosa
import pickle
import random
from sklearn.preprocessing import StandardScaler


class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = int(sr / 1000 * max_ms)


        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:, :max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)  # the padding is randomly distributed around the sig
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

    # ----------------------------
    # !!! Not sure if want this - no wrapping but shift maybe useful.
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, spec_type, n_mels=64, n_fft=512, hop_length=None):
        # Wang et al use n_fft=441 (20ms at 22050Hz), smaller size = better freq rez.
        if hop_length is None:
            hop_length = n_fft // 2
        sig, sr = aud
        if spec_type == 'Mel':
            # spec has shape [channel, n_mels, dur], where dur = num_samples / n_fft//hop ratio
            # the dur is the number of SFTF, and depends on the fft window size and hop size
            # eg. num_samples = 22050*0.2 = 4410
            # 4410 / (256//2) = 35
            spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)(sig)
            # Convert to decibels
            spec = torchaudio.transforms.AmplitudeToDB()(spec)

            return spec

        elif spec_type == 'CQT':
            sig = sig.numpy()
            ''' TO DO: my_window = np.kaiser(512, 4.0) or something like this?
                librosa.filters.get_window('hann', 512)'''
            C = np.abs(librosa.cqt(sig, sr=sr, n_bins=n_fft // 2 + 1, hop_length=hop_length))
            sgram = librosa.amplitude_to_db(C, ref=np.max)
            spec = torch.from_numpy(sgram)

            return spec

    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # ----------------------------
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except FileExistsError as e:
        print(e)
        answer = input(" WARNING! This will overwrite the current args files in this folder. "
                       "Please make sure the spectrogram parameters are the same."
                       "\n Do you want to continue? [y/n]:")
        if answer == 'n':
            sys.exit()


# ----------------------------
# Make the spectrograms
# ----------------------------
def make_specs(audio_file, s_type, duration, n_mels, n_fft, hop_length, shift=False, mask=False):
    duration = duration
    shift_pct = 0.1

    aud = AudioUtil.open(str(audio_file))
    # Some sounds have a higher sample rate, or fewer channels compared to the
    # majority. So make all sounds have the same number of channels and same
    # sample rate. Unless the sample rate is the same, the pad_trunc will still
    # result in arrays of different lengths, even though the sound duration is
    # the same.
    new_aud = AudioUtil.pad_trunc(aud, duration)
    if shift:
        new_aud = AudioUtil.time_shift(new_aud, shift_pct)
    sgram = AudioUtil.spectro_gram(new_aud, s_type, n_mels, n_fft=n_fft, hop_length=hop_length)
    if mask:
        sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        return sgram

    return sgram


def make_specs_train_test_split():
    AUDIO_DIR = args.audio_directory
    DATA_DEST = args.data_destination

    SHIFT = args.shift  # False
    MASK = args.mask  # False
    N_MELS = args.mels  # 64
    N_FFT = args.fft  # 256
    SR = args.samplerate  # 22050
    S_TYPE = args.SpecType
    DURATION = args.duration  # 200
    HOP = args.hop_length  # 128

    test_path = os.path.join(DATA_DEST, 'test_data/')
    train_path = os.path.join(DATA_DEST, 'train_data/')

    make_dir(test_path)
    make_dir(train_path)

    # get total num of file (for showing progress)
    total = len(next(os.walk(AUDIO_DIR))[2])
    i = 1

    # get values for train/test split
    total_patients = 41  # MUST BE HARD CODED!!
    num_train_samples = round(0.8 * total_patients)

    # get random list of patients for train and test
    training_patient_ids = random.sample(range(0, total_patients), num_train_samples)

    tensor_shape_list = []

    # loop through audio dir, make spectrogram, and save to test or train
    for filename in os.listdir(AUDIO_DIR):
        file_path = os.path.join(AUDIO_DIR, filename)
        file_stem = Path(file_path).stem
        patient_id = int(file_stem.split('_')[1])  # take the second val in the file name which indicates the patient
        # make the spectrogram (tensor dtype)
        spec = make_specs(file_path, S_TYPE, DURATION, N_MELS, N_FFT, HOP, shift=SHIFT, mask=MASK)

        # for double checking the tensors are all the same shape
        if spec.shape not in tensor_shape_list:
            tensor_shape_list.append(spec.shape)

        if patient_id in training_patient_ids:  # 80/20 train/test split ISH
            new_file_path = os.path.join(train_path, file_stem)
        else:
            new_file_path = os.path.join(test_path, file_stem)

        torch.save(spec, f"{new_file_path}.pt")
        print(f"File [{i}/{total}] --- Converted {file_path} to {S_TYPE}-spectrogram, saved to {new_file_path}")
        i += 1

    # Give the total num of test and train files
    test_total = len(next(os.walk(test_path))[2])
    train_total = len(next(os.walk(train_path))[2])
    print(f'Test files: {test_total} \nTrain files: {train_total} \nChecking tensor shapes: {tensor_shape_list}')


def make_specs_train_test_split_rand():
    AUDIO_DIR = args.audio_directory
    DATA_DEST = args.data_destination

    SHIFT = args.shift  # False
    MASK = args.mask  # False
    N_MELS = args.mels  # 64
    N_FFT = args.fft  # 256
    SR = args.samplerate  # 22050
    S_TYPE = args.SpecType
    DURATION = args.duration  # 200
    HOP = args.hop_length

    test_path = os.path.join(DATA_DEST, 'test_data/')
    train_path = os.path.join(DATA_DEST, 'train_data/')

    make_dir(test_path)
    make_dir(train_path)

    # get total num of file (for showing progress)
    total = len(next(os.walk(AUDIO_DIR))[2])
    i = 1

    # loop through audio dir, make spectrogram, and save to test or train
    for filename in os.listdir(AUDIO_DIR):
        file_path = os.path.join(AUDIO_DIR, filename)
        file_stem = Path(file_path).stem
        # make the spectrogram (tensor dtype)
        spec = make_specs(file_path, S_TYPE, DURATION, N_MELS, N_FFT, HOP, shift=SHIFT, mask=MASK)
        if np.random.rand(1) < 0.8:  # 80/20 train/test split BUT NOT 100% ACCURATE!!!
            new_file_path = os.path.join(train_path, file_stem)
        else:
            new_file_path = os.path.join(test_path, file_stem)

        torch.save(spec, f"{new_file_path}.pt")
        print(f"File [{i}/{total}] --- Converted {file_path} to {S_TYPE}-spectrogram, saved to {new_file_path}")
        i += 1

    # Give the total num of test and train files
    test_total = len(next(os.walk(test_path))[2])
    train_total = len(next(os.walk(train_path))[2])
    print(f'Test files: {test_total} \n Train files: {train_total}')


def make_specs_dir():
    AUDIO_DIR = args.audio_directory
    DATA_DEST = args.data_destination
    FOLDER = args.folder

    SHIFT = args.shift  # False
    MASK = args.mask  # False
    N_MELS = args.mels  # 64
    N_FFT = args.fft  # 256
    SR = args.samplerate  # 22050
    S_TYPE = args.SpecType
    DURATION = args.duration  # 200
    HOP = args.hop_length

    folder_path = os.path.join(DATA_DEST, FOLDER)
    make_dir(DATA_DEST)
    make_dir(folder_path)

    # get total num of file (for showing progress)
    total = len(next(os.walk(AUDIO_DIR))[2])
    i = 1

    # loop through audio dir, make spectrogram, and save to single dir
    for filename in os.listdir(AUDIO_DIR):
        file_path = os.path.join(AUDIO_DIR, filename)
        file_stem = Path(file_path).stem
        # make the spectrogram (tensor dtype)
        spec = make_specs(file_path, S_TYPE, DURATION, N_MELS, N_FFT, HOP, shift=SHIFT, mask=MASK)
        new_file_path = os.path.join(folder_path, file_stem)

        torch.save(spec, f"{new_file_path}.pt")
        print(f"File [{i}/{total}] --- Converted {file_path} to {S_TYPE}-spectrogram, saved to {new_file_path}")
        i += 1

    # Give the total num of test and train files
    total = len(next(os.walk(folder_path))[2])
    print(f'Files converted: {total}')


def process_commandline():
    parser = argparse.ArgumentParser(
        description='To create train/val/test folders of spectrogram data')

    # basic parameter arguments
    parser.add_argument('audio_directory', default="../recordings",
                        help="Directory containing dataset wavs")
    parser.add_argument('data_destination', default="../spec_data/Spectrograms",
                        help="Parent directory to contain spectrogram data")
    parser.add_argument('--folder', default="data",
                        help="Folder name to contain spectrogram data (Use this if NOT splitting)")

    # Optional arguments for data:
    parser.add_argument('--split', action="store_true", default=False,
                        help="split train and test data")
    parser.add_argument('--split_rand', action="store_true", default=False,
                        help="split train and test data ranomly")
    parser.add_argument('--SpecType', '-sp', action="store", default='Mel', choices=['Mel', 'CQT'],
                        help="The type of spectrogram used")
    parser.add_argument('--mask', '-ma', action="store_true", default=False,
                        help="Add masking to the data")
    parser.add_argument('--shift', '-sh', action="store_true", default=False,
                        help="Enable shifting audio")
    parser.add_argument('--mels', '-me', default=64, type=int,
                        help="Define the number of freq bins")
    parser.add_argument('--fft', '-f', default=350, type=int,
                        help="Define the length of the fft window in samples")
    parser.add_argument('--samplerate', '-sa', default=22050, type=int,
                        help="Define the sample rate")
    parser.add_argument('--duration', '-du', default=200, type=int,
                        help="Define duration for padding")
    parser.add_argument('--hop_length', '-hl', default=116, type=int,
                        help="Define hop length of sftf in samples")
    global args
    args = parser.parse_args()

    # Must have data directory and run title
    if not args.audio_directory or not args.data_destination:
        parser.error('Must supply both audio directory and data destination')

    return args


# SHIFT = args.shift  # False
# MASK = args.mask  # False
# N_MELS = args.mels  # 64
# N_FFT = args.fft  # 256
# SR = args.samplerate  # 22050
# S_TYPE = args.SpecType  # Mel


def main():
    process_commandline()
    DATA_DEST = args.data_destination
    SPLIT = args.split
    SPLIT_RAND = args.split_rand

    # make test, train directories and args files
    args_txt = os.path.join(DATA_DEST, 'args.txt')
    args_pkl = os.path.join(DATA_DEST, 'args.pkl')

    if SPLIT:
        print('--- Making test and train data split (with no patient overlaps) ---')
        make_specs_train_test_split()
    elif SPLIT_RAND:
        print('--- Making test and train data split (patient overlaps possible) ---')
        make_specs_train_test_split_rand()
    else:
        print('--- Making single dir of spec data ---')
        make_specs_dir()

    with open(args_txt, 'w+') as f:
        f.write(str(args))
    with open(args_pkl, 'wb+') as f:
        pickle.dump(vars(args), f)


if __name__ == '__main__':
    main()
