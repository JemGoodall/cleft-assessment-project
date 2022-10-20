'''
Helper functions
'''

import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import pickle
import pandas as pd
import sys
from sklearn.metrics import mean_absolute_error
import csv
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from model import AudioClassifier, weights_init, LRCN

# ----------------------------
# Save the model - use 'ver' to save under diff names
# ----------------------------
def saveModel(RUN_TITLE, model):
    model_name = RUN_TITLE + '_model.pt'
    model_path = os.path.join('final_exp', RUN_TITLE, model_name)
    torch.save(model.state_dict(), model_path)

# ----------------------------
# Loading saved model
# ----------------------------
def load_model(RUN_TITLE, input_dims, hidden_dims, lrcn=False, ):
    model_name = RUN_TITLE + '_model.pt'
    model_path = os.path.join('final_exp', RUN_TITLE, model_name)
    kwargs, state = torch.load(model_path)
    if lrcn:
        myModel = LRCN(input_dims, hidden_dims)
    else:
        myModel = AudioClassifier()
    myModel.load_state_dict(state)
    myModel = myModel.to(DEVICE)
    return myModel

# ----------------------------
# write full training output to file
# ----------------------------
def write_training_history_to_file(history, RUN_TITLE, total_time_sec, time_per_epoch_sec, epochs):
    with open('log_train.csv', 'a', newline='') as f:
        print('Writing results to log_train.csv')
        writer = csv.writer(f)
        writer.writerow([RUN_TITLE])
    pd.DataFrame(history).to_csv('./log_train.csv', index=False, mode='a', sep='|')

    with open('log_train_summaries.csv', 'a', newline='') as file:
        print('Writing results to log_train_summaries.csv')
        writer = csv.writer(file, delimiter='|')
        n = epochs - 1
        writer.writerow([RUN_TITLE, n + 1, history["loss"][n], history["acc"][n],
                         history["val_loss"][n], history["val_acc"][n],
                         f'{total_time_sec / 60:5.2f} ({time_per_epoch_sec / 60:5.2f}) min'])
# ----------------------------
# time_per_epoch
# ----------------------------
def print_time_per_epoch(start_time_sec, end_time_sec, epochs):
    total_time_sec = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print(f'Time total:     {total_time_sec / 60:5.2f} min')
    print(f'Time per epoch: {time_per_epoch_sec / 60:5.2f} min')
    return total_time_sec, time_per_epoch_sec

# ----------------------------
# Plot graph
# ----------------------------
def plot_results(history, epochs, RUN_TITLE, model_path):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.suptitle(RUN_TITLE)

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.title('Training and Validation Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.title('Training and Validation Loss')
    # plt.show()

    # Save graph
    plt.savefig(os.path.join(model_path, 'Acc_Loss_graph.png'))
    plt.clf()


# ----------------------------
# Get mean and std of training set
# ----------------------------
def get_mean_and_std(train_set, norm_type="mels"):
    # load the data into one big batch and get the mean and variance using torch methods
    # requires small dataset to fit into memory
    loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set))
    data = next(iter(loader))
    # data is [sgram_data, classes, relative paths]
    # sgram_data = torch.Size([num_train_samples, channels(1), mels, time])
    if norm_type == "mels":
        # get mean and std for each mel across the data set (i.e. avg across N, C, and W)
        mean = data[0].mean((0, 1, 3)).view(1, -1, 1)  # mean.shape = torch.Size([1, mels, 1])
        std = data[0].std((0, 1, 3)).view(1, -1, 1)  # because transforms.Normalise works per sample: ([C, H, W])
        print(f'Normalising by mels. Mean shape = {mean.shape}')
    else:
        # get global mean and std
        mean, std = data[0].mean(), data[0].std()
        print(f'normalising globally. Mean and std = {mean, std}')
        
    return mean, std


# ----------------------------
# Save mean and std of training set
# ----------------------------
def save_mean_and_std(mean, std, RUN_TITLE):
    scalers = [mean, std]
    pickle_path = os.path.join('final_exp', RUN_TITLE, 'scalers.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(scalers, f)


# ----------------------------
# Create model directory path
# ----------------------------
def make_model_dir_path(RUN_TITLE):
    model_dir = os.path.join('final_exp', RUN_TITLE)
    return model_dir


# ----------------------------
# Analyse results
# ----------------------------
def analyse_results(df1):
    """
    :argument: df with columns: predictions|targets|file_stem
                where file_stem is formatted as '[class]_[patientName]_[idx].pt'
                and represents the splits of the original audio file
    :return
        mae: int, mean absolute error per file
        total: int, num_of_files
        per_file_df: df with columns file_stem|avg_preds|targets|error
    """
    df = df1.copy()
    df["file_stem"] = df["file_stem"].replace(r'_([\d])+\.pt', '', regex=True)
    df["file_stem"] = [f'{x[0]}_{x[1]}_{x[2]}' for x in df['file_stem'].str.split('_')]
    per_file_df = df.groupby(['file_stem']).mean()
    per_file_df.columns = per_file_df.columns.str.replace('predictions', 'avg_preds')
    per_file_df['error'] = per_file_df.apply(lambda x: x['avg_preds'] - x['targets'], axis=1)
    mae = mean_absolute_error(y_true=per_file_df["targets"], y_pred=per_file_df["avg_preds"])
    total = per_file_df.shape[0]
    return mae, total, per_file_df


# ----------------------------
# write results
# ----------------------------
def write_summmary(test_acc, total_prediction, mae, total_file, f1, RUN_TITLE):
    with open('./log_test_summaries.csv', 'a', newline='') as f:
        print('Writing results to log_test_summaries.csv')
        writer = csv.writer(f)
        writer.writerow([RUN_TITLE, test_acc, total_prediction, mae, total_file, f1])


def write_full_summary(df, RUN_TITLE, per_file=True):
    if per_file:
        run_title = RUN_TITLE + '_perFile'
    else:
        run_title = RUN_TITLE
    path = f'final_exp/{RUN_TITLE}/results_{run_title}.csv'
    print(f'Writing FULL results to {path}')
    df.to_csv(path, mode='w', sep='|')


# ----------------------------
# get prediction list - correct predictions are within +-1 of target.
# ----------------------------
def get_correct_list(targets, predictions):
    output = []
    for item in list(zip(targets, predictions)):
        if item[0] > item[1] + 1 or item[0] < item[1] - 1:
            output.append(item[1])
        else:
            output.append(item[0])
    return torch.tensor(output)


# ----------------------------
# plot ROC curve for each class
# ----------------------------
def plot_roc_curve(y_test, y_pred, save_path='ROC_curve.png'):
    n_classes = len(np.unique(y_test))
    y_test = label_binarize(y_test, classes=np.arange(n_classes))
    y_pred = label_binarize(y_pred, classes=np.arange(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    # plt.figure(figsize=(10,5))
    plt.figure(dpi=600)
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
             color="deeppink", linestyle=":", linewidth=4, )

    plt.plot(fpr["macro"], tpr["macro"],
             label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
             color="navy", linestyle=":", linewidth=4, )

    colors = cycle(["aqua", "darkorange", "darkgreen", "yellow", "blue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]), )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) curve")
    plt.legend()

    plt.savefig(save_path)
    plt.clf()


# ----------------------------
#  create a scatter plot for per patient predictions.
#  takes a DF with columns predictions, targets, file_stem
# ----------------------------
def scatter_plot_per_patient(df1, save_path='bubble_plot.png'):
    #https://datascience.stackexchange.com/questions/89692/plot-two-categorical-variables
    df = df1.copy()
    df["file_stem"] = [f'ID{x[1]}_Class{x[0]}' for x in df['file_stem'].str.split('_')]
    counts = df.groupby(['predictions', 'targets', 'file_stem']).size().reset_index(name='Count')
    xnoise, ynoise = np.random.random(len(counts)) / 2, np.random.random(len(counts)) / 2
    counts["targets"] = counts['targets'] + xnoise
    counts["predictions"] = counts['predictions'] + ynoise

    # Plot the scatterplot
    ax = sns.scatterplot(data=counts,
                  x='predictions', y='targets', hue='file_stem', size='Count', edgecolors="black", sizes=(50, 1100), alpha=0.9,palette="Paired")
    ax.invert_yaxis()
    plt.xticks([0.25, 1.25, 2.25, 3.25, 4.25], [0,1,2,3,4])  # The reason the xticks start at 0.25
    # and go up in increments of 1 is because the center of the noise will be around 0.25 and ordinal
    # encoded labels go up in increments of 1.
    plt.yticks([0.25, 1.25, 2.25,  3.25, 4.25], [0,1,2,3,4])  # This has the same reason explained for xticks
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Classification per patient")

    # Format figure size, spines and grid
    ax.grid(axis='y', which='minor', color='black', alpha=0.2)
    ax.grid(axis='x', which='minor', color='black', alpha=0.2)
    sns.despine(left=True)
    # Format ticks
    ax.tick_params(axis='both', length=0, pad=10, labelsize=12)
    ax.tick_params(axis='x', which='minor', length=25, width=0.8, color=[0, 0, 0, 0.2])
    minor_xticks = [tick + 0.5 for tick in ax.get_xticks() if tick != ax.get_xticks()[-1]]
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_xticks, minor=True)

    ax.legend(bbox_to_anchor=(1.3, 1), borderaxespad=0)
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()
    # plt.show()


# def save_experiment_details(model, single_dataset_sample, args, history, num_epochs, RUN_TITLE):
#     final_epoch = num_epochs+1
#     info_file_path = os.path.join('experiments', RUN_TITLE, 'details.txt')
#     model_txt_info = summary_string(model, input_size=(single_dataset_sample[0], single_dataset_sample[1], single_dataset_sample[2]))
#     with open(info_file_path, 'w+') as f:
#         f.write(f'================================================================ \n '
#               f'TRAINING {RUN_TITLE}'
#               f'\n================================================================\n')
#         f.write(model_txt_info)
#         f.write('\n')
#         f.write(args)
#         f.write('\n')
#         f.write(f'Final epoch results: Train_loss={history["loss"][final_epoch]}, train+ac{history["acc"][final_epoch]}, val_loss={history["val_loss"][final_epoch]}, val_acc={history["val_acc"][final_epoch]}')


# ''' Make image from spec data '''
# # (used in the `plot_classes_preds` function below)
# def _matplotlib_imshow(spec, one_channel=False):
#     spec = spec / 2 + 0.5     # unnormalize
#     npimg = spec.numpy()
#     plt.imshow(npimg, cmap="Greys")
#
#
# def plot_save_spectrogram(specgram, graph_type='frames', title=None):
#     fig, axs = plt.subplots(1, 1)
#     if graph_type == 'frames':
#         axs.set_title(title or "Spectrogram (db)")
#         axs.set_ylabel("freq_bin")
#         axs.set_xlabel("frame")
#         im = axs.imshow(specgram, origin="lower", aspect="auto")
#         fig.colorbar(im, ax=axs)
#
#     elif graph_type == 'time':
#         img = librosa.display.specshow(specgram.numpy(), sr=sr, x_axis='time', y_axis='cqt_note', ax=axs)
#         axs.set_title(title)
#         fig.colorbar(img, ax=axs, format="%+2.0f dB")
#
#     plt.imshow()

# ----------------------------
# plot ROC curve for each class
# ----------------------------
def plot_roc_curve_NEW(y_test, y_pred, save_path='ROC_curve.png'):
    n_classes = 5
    y_test = label_binarize(y_test, classes=np.arange(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    # plt.figure(figsize=(10,5))
    plt.figure(dpi=600)
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
             color="deeppink", linestyle=":", linewidth=4, )

    plt.plot(fpr["macro"], tpr["macro"],
             label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
             color="navy", linestyle=":", linewidth=4, )

    colors = cycle(["aqua", "darkorange", "darkgreen", "yellow", "blue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]), )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) curve")
    plt.legend()

    plt.savefig(save_path)
    plt.clf()