"""
Script which takes a model and test audio data
Makes spectrograms using the parameters provided
Uses model provided to predict labels.
Puts all info into a master dir

Run as:
python test_wav_files.py [MASTER_DIR_NAME] [AUDIO_DIR] [path_to_model] [path_to_pkl_spec_params]
"""

import os
import sys
import argparse
from make_spec_data import make_specs, AudioUtil
import pickle
from pathlib import Path
import torch
from test import make_data_loaders, inference, analyse_results, write_summmary, write_full_summary
from train import AudioClassifier
import csv


parser = argparse.ArgumentParser(description='to test a dir of audio files')

# basic parameter arguments
parser.add_argument('master_dir', default="MASTER_DIR",
                    help="Directory to contain all the info")
parser.add_argument('audio_files',  default="../recordings",
                    help="dir of audio files")
parser.add_argument('model_path', default="../models/test_model.pt",
                    help="Path to the model")
parser.add_argument('spec_params', default=' ',
                    help="path to pickled dict of spec params")

args = parser.parse_args()
MASTER_DIR = args.master_dir
AUDIO_FILES = args.audio_files
MODEL_PATH = args.model_path
SPEC_PARAMS_PATH = args.spec_params


TEST = 'test'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
0. set up MASTER_dir and sub dirs to store all the files
"""


def make_dirs(MASTER_PATH):
    split_path = os.path.join(MASTER_PATH, 'split_audio')
    spec_path = os.path.join(MASTER_PATH, 'spec_data')
    try:
        os.makedirs(MASTER_PATH)
        os.makedirs(split_path)
        os.makedirs(spec_path)
    except FileExistsError as e:
        print(e)
        answer = input("Do you want to continue? [y/n]:")
        if answer == 'n':
            sys.exit()

    return split_path, spec_path


"""
1. ---- preprocess and split audio files into frames ----
    input: dir of audio files
    output: new_dir of split audio files
"""


def split_audio():
    # TO DO (import from another file?)
    pretend_split_audio_path = AUDIO_FILES
    return pretend_split_audio_path


"""
2. ---- make spectrograms ----
    input: dir of split audio files / list of args of spec / spec_path
    output: dir of spectrograms with same names
"""


def make_spectrograms(split_audio_path, SPEC_PARAMS_PATH, spec_path):
    with open(SPEC_PARAMS_PATH, 'rb') as handle:
        p = pickle.load(handle)

    # get total num of file (for showing progress)
    total = len(next(os.walk(split_audio_path))[2])
    i = 1

    for filename in os.listdir(split_audio_path):
        file_path = os.path.join(split_audio_path, filename)
        file_stem = Path(file_path).stem
        # make the spectrogram (tensor dtype)
        spec = make_specs(file_path, p['SpecType'], p['samplerate'], p['mels'], p['fft'], shift=p['shift'],
                          mask=p['mask'])
        new_file_path = os.path.join(spec_path, file_stem)

        torch.save(spec, f"{new_file_path}.pt")
        print(f"File [{i}/{total}] --- Converted {file_path} to {p['SpecType']}-spectrogram, saved to {new_file_path}")
        i += 1


"""
3. ---- load data and model ----
    input: path to spectrograms, test, batch_size, model_path
    output: test_ds, test_dl
"""
def load_data(spec_path, TEST, batch_size=16):
    test_ds, test_dl = make_data_loaders(spec_path, TEST, batch_size)
    return test_ds, test_dl,

"""
4. ---- load and test model ----
    input: path to model
    output loaded model
"""
def load_model(MODEL_PATH):
    myModel = AudioClassifier()
    myModel.load_state_dict(torch.load(MODEL_PATH))
    myModel = myModel.to(DEVICE)
    # Check that it is on Cuda
    # print(f"Model is on: {next(myModel.parameters()).DEVICE}")

    return myModel

"""
5. ---- inference and analysis ---
    input: model, test_dl
    output: csv file of test results
"""
def run_model(model, test_dl):
    test_acc, total_prediction, df = inference(model, test_dl, DEVICE)
    mse, total_file, per_file_df = analyse_results(df)
    model_ver = Path(MODEL_PATH).stem
    log_path = os.path.join(MASTER_DIR, f'./log_{model_ver}.csv')

    with open(log_path, 'a+', newline='') as f:
        print(f'Writing results to {log_path}')
        writer = csv.writer(f)
        writer.writerow(['model_ver', 'acc_per_frame', 'total_frames', 'mse_per_file', 'num_files'])
        writer.writerow([model_ver, test_acc, total_prediction, mse, total_file])
        writer.writerow(['---'])
    per_file_df.to_csv(log_path, mode='a', sep='|')
    print(f'Test accuracy per frame: {test_acc:.2f}, Total test items: {total_prediction}')
    print(f'MSE for files: {mse}, Total num of files: {total_file}')

def main():
    split_path, spec_path = make_dirs(MASTER_DIR)
    split_audio_path = split_audio()
    make_spectrograms(split_audio_path, SPEC_PARAMS_PATH, spec_path)
    _, test_dl = load_data(spec_path, TEST, batch_size=16)
    myModel = load_model(MODEL_PATH)
    run_model(myModel, test_dl)

if __name__ =='__main__':
    main()