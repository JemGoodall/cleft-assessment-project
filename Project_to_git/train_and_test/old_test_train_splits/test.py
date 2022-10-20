"""
 --- TESTING ---
run as:
python test.py [data_directory] [model_path] [run_title]
e.g:
python test.py ../spec_data/testing_script/test_data ../models/test_model.pt trail_run

expects 'scalers.pkl' file in the parent data dir
"""
import sys

import make_dataset as ds
from model import AudioClassifier
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchsummary import summary
import csv
import os
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
from torchvision import transforms
from torchmetrics.functional import f1_score



def _get_mean_and_std(MODEL_PATH):
    parent_path = str(Path(MODEL_PATH).parent)
    pickle_path = os.path.join(parent_path, 'scalers.pkl')
    with open(pickle_path, 'rb') as f:
        scalers = pickle.load(f)

    return scalers  # a list: [mean, std]


def make_data_loaders(DATA_PATH, BATCH_SIZE=16, NORM_TYPE=None):
    # get meta data from the data set
    df = ds.get_df(DATA_PATH)
    print(f'----- Meta data sample ---- \n {df.head()}')

    if NORM_TYPE:
        mean, std = _get_mean_and_std(DATA_PATH)
        print(f'Mean and std (single vals): {mean.shape}')

        test_ds = ds.SoundDS(df, DATA_PATH, transform=transforms.Normalize(mean, std))
    else:
        test_ds = ds.SoundDS(df, DATA_PATH)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    return test_ds, test_dl


# ----------------------------
# Loading saved model
# ----------------------------
def load_model(MODEL_PATH):
    myModel = AudioClassifier()
    myModel.load_state_dict(torch.load(MODEL_PATH))
    myModel = myModel.to(DEVICE)
    return myModel


# ----------------------------
# Inference
# ----------------------------
def inference(model, test_dl, DEVICE):
    model.eval()
    correct_prediction = 0
    total_prediction = 0
    history = []
    all_predictions = []
    all_targets = []

    # Disable gradient updates
    with torch.no_grad():
        for data in test_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels, files = data[0].to(DEVICE), data[1].to(DEVICE), data[2]

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            all_predictions.append(prediction)
            all_targets.append(labels)

            # append history to make into df later
            for i, clss in enumerate(labels):
                # print(f'(pred={prediction[i]}, tgt={clss}, input={file_ids[i]})', end="")
                history.append([prediction[i].item(), clss.item(), files[i]])

            # pause = input('Wait')
            # if pause:
            #     sys.exit()
            # idea for per frame:
            # torch.max(outputs) returns [float, float, float], [class, class, class]
            # can take avg of classes over all frames

        # make data frame of all predictions, targets, files
        df = pd.DataFrame(history, columns=['predictions', 'targets', 'file_stem'])
        test_acc = correct_prediction / total_prediction
        all_predictions = torch.cat(all_predictions).cpu()  # get long 1Dtensor of all preds
        all_targets = torch.cat(all_targets).cpu()
        print(all_predictions, all_targets)
        print(all_predictions.shape, all_targets.shape)
        sys.exit()
        f1 = f1_score(all_predictions, all_targets, num_classes=5, average='macro').item()  # f1_score returns a tensor, use item()

        # print(df.head())
        print(classification_report(all_predictions, all_targets, target_names=['class 0', 'class 1', 'class 2', 'class 3', 'class 4']))
        ConfusionMatrixDisplay.from_predictions(all_targets, all_predictions, labels=[0, 1, 2, 3, 4], cmap=plt.cm.Blues)
        plt.savefig(f'../test_results_confusion_matrix/{args.run_title}.png')

        return test_acc, f1, total_prediction, df


# ----------------------------
# Analyse results
# ----------------------------
def analyse_results(df):
    """
    :argument: df with columns: predictions|targets|file_stem
                where file_stem is formatted as '[class]_[patientName]_[idx].pt'
                and represents the splits of the original audio file
    :return
        mse: int, mean squared error per file
        total: int, num_of_files
        per_file_df: df with columns file_stem|avg_preds|targets|error
    """
    df["file_stem"] = df["file_stem"].replace(r'_([\d])+\.pt', '', regex=True)
    per_file_df = df.groupby(['file_stem']).mean()
    per_file_df.columns = per_file_df.columns.str.replace('predictions', 'avg_preds')
    per_file_df['error'] = per_file_df.apply(lambda x: x['avg_preds'] - x['targets'], axis=1)
    mse = mean_squared_error(per_file_df["avg_preds"], per_file_df["targets"], squared=True)
    total = per_file_df.shape[0]
    return mse, total, per_file_df


# ----------------------------
# write results
# ----------------------------
def write_summmary(test_acc, total_prediction,mse, total_file, f1):
    RUN_TITLE = args.run_title
    MODEL_PATH = args.model_path

    with open('./log_test_summaries.csv', 'a', newline='') as f:
        print('Writing results to log_test_summaries.csv')
        writer = csv.writer(f)
        model_ver = Path(MODEL_PATH).stem
        writer.writerow([RUN_TITLE, model_ver, test_acc, total_prediction, mse, total_file, f1])

def write_full_summary(df, per_file=True):
    RUN_TITLE = args.run_title

    if per_file:
        run_title = RUN_TITLE + '_perFile'
    else:
        run_title = RUN_TITLE
    path = f'../test_results/log_{run_title}.csv'
    print(f'Writing FULL results to {path}')
    df.to_csv(path, mode='a', sep='|')


# ----------------------------
# process command line args
# ----------------------------
def process_commandline():
    parser = argparse.ArgumentParser(
        description='To test the model')

    # basic parameter arguments
    parser.add_argument('data_directory', default="../spec_data/testing_script/test_data",
                        help="Directory to contain spectrogram data")
    parser.add_argument('model_path', default="../models/test_model.pt",
                        help="Path to the model")
    parser.add_argument('run_title', nargs='?', default='testing X',
                        help="Set the run title (saves to graph and logs)")
    parser.add_argument('--test', '-t', action="store_true", default=False,
                        help="Whether this is a test (changes meta_data action)")

    # optional args
    parser.add_argument('--batchsize', '-b', action="store", default=32, type=int,
                        help="Define the batch size")
    parser.add_argument('--save_predictions', '-s', action="store_true", default=False,
                        help="Save all the predictions to a csv file")

    global args
    args = parser.parse_args()

    # Must have data directory and run title
    if not args.data_directory or not args.run_title or not args.model_path:
        parser.error('Must supply, data directory, model path, and run title.')

    return args



# DATA_DIR = args.data_directory  # ../spec_data/testing_script/test_data
# MODEL_PATH = args.model_path  # ../model/test_model.pt
# RUN_TITLE = args.run_title  # testing X
# TEST = args.test  # False
#
# BATCH_SIZE = args.batchsize  # 32
# SAVE_PREDS = args.save_predictions  # False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    process_commandline()
    RUN_TITLE = args.run_title
    DATA_DIR = args.data_directory
    BATCH_SIZE = args.batchsize
    SAVE_PREDS = args.save_predictions
    MODEL_PATH = args.model_path

    print(f'================================================================ \n '
          f' TESTING {RUN_TITLE}'
          f'\n================================================================')
    # Creating data loaders
    print("Making Data loaders....")
    test_ds, test_dl = make_data_loaders(DATA_DIR, BATCH_SIZE)

    # Load model and put on GPU
    myModel = load_model(MODEL_PATH)
    size = test_ds[2][0].shape
    summary(myModel, input_size=(size[0], size[1], size[2]))  # channels, H, W

    print("----- Acc and F1 scores  -----")
    test_acc, f1, total_prediction, df = inference(myModel, test_dl, DEVICE)
    print("----- Sample predictions -----")
    mse, total_file, per_file_df = analyse_results(df)
    print(per_file_df.head())
    print(f'Test accuracy and macro f1 per frame: {test_acc:.2f} {f1:.2f}, Total test items: {total_prediction}')
    print(f'MSE for files: {mse}, Total num of files: {total_file}')

    # Save short summary of test run and of per_file results
    write_summmary(test_acc, total_prediction, mse, total_file, f1)
    write_full_summary(per_file_df, per_file=True)

    # Save all predictions
    if SAVE_PREDS:
        write_full_summary(df)



if __name__ == '__main__':
    main()
