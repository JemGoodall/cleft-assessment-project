# https://stackoverflow.com/questions/59584457/pytorch-is-there-a-definitive-training-loop-similar-to-keras-fit
# https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

'''
Saves all epoch results to log_train.csv  
Saves the final epoch run to log_train_summaries.csv
Saves model every 10 epochs, and best model to ../models/[RUN_TITLE] ... 
Save data norm transform scalers to the data dir under scalers.pkl

run as:
python train.py INPUT_SPEC_DATA run_title 
'''
import sys

import make_dataset as ds
from args import process_commandline
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchvision import transforms
from torchsummary import summary
# import csv
import os
import time
import pickle
from pathlib import Path
from helper_funcs import *
# from model import AudioClassifier, weights_init, LRCN
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics.functional import f1_score
print('WARNING! RUNNING ON OLD TRAIN_TEST FILE!!!')
torch.manual_seed(0)  # reproducibility!
'''GLOBALS '''
N_CLASSES = 5
SEED = 6  # avoid 5
# Weights = [0.8598, 0.9346, 0.8422, 0.7803, 0.5831]  # unbalanced dataset
# Weights = [0.8041, 0.8626, 0.7788, 0.8462, 0.7083]  # balanced dataset
# Weights = [2.978, 6.3712, 2.638, 1.8967, 1.0]  # korin's method:  num_chunks_in_largest_class)/(num_chunks_in_class_k)
Weights = [1.7868, 3.8227, 1.5828, 1.138, 1.0]   # korin's method but scaled down by 0.6 (excluding class 4)
class_weights = torch.FloatTensor(Weights).cuda()


# Each batch has a shape of (batch_sz, num_channels, Mel freq_bands, time_steps) E.G. torch.Size([32, 1, 64, 35])

history = {'epoch': [], 'loss': [], 'acc': [], 'val_loss': [],
           'val_acc': []}  # Collects per-epoch loss and acc like Keras' fit().


# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, val_dl, num_epochs, lr, RUN_TITLE, device, weights=False, lrcn=False):
    # Loss Function, Optimizer and Scheduler
    if not weights:
        criterion = nn.CrossEntropyLoss()  # applies nn.LogSoftmax() and nn.NLLLoss()
        print('no weights applied')
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f'class weights applied: {Weights}')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr / 10,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    # Repeat for each epoch
    for epoch in range(num_epochs):

        # --- TRAIN ON TRAINING SET -------------------------------------
        # Repeat for each batch in the training set
        model.train()
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if lrcn:
                inputs = (inputs, data[3])  # include lengths for LRCN model
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()  # only in train
            optimizer.step()  # only in train
            scheduler.step()  # only in train

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_running_loss = 0.0
        val_correct_prediction = 0
        val_total_prediction = 0

        with torch.no_grad():
            for i, data in enumerate(val_dl):
                # Get the input features and target labels, and put them on the GPU
                inputs, labels = data[0].to(device), data[1].to(device)

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                if lrcn:
                    inputs = (inputs, data[3])  # include lengths for LRCN model

                # forward
                val_outputs = model(inputs)  # outputs torch.Size([batchsize, classes])
                loss = criterion(val_outputs, labels)

                # Keep stats for Loss and Accuracy
                val_running_loss += loss.item()

                # Get the predicted class with the highest score
                _, prediction = torch.max(val_outputs, 1)  # max over classes (dim=1),
                # predictions is a tensor with the indexes of the highest values in dim=1
                # Count of predictions that matched the target label
                val_correct_prediction += (prediction == labels).sum().item()
                val_total_prediction += prediction.shape[0]

        # Print stats at the end of the epoch
        num_batches = len(val_dl)
        val_loss = val_running_loss / num_batches
        val_acc = val_correct_prediction / val_total_prediction

        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:  # only report every 10 epochs
            print(f'Epoch: {epoch + 1}/{num_epochs} | Loss: {avg_loss:.3f}, Accuracy: {acc:.3f} |'
                  f' Val_loss: {val_loss:.3f}, Val_accuracy: {val_acc:.3f}')

        history['loss'].append(round(avg_loss, 4))
        history['acc'].append(round(acc, 4))
        history['val_loss'].append(round(val_loss, 4))
        history['val_acc'].append(round(val_acc, 4))
        history['epoch'].append(epoch + 1)

        # Save model every 20 epochs and at the final epoch
        if (epoch + 1) % 20 == 0 or (epoch + 1) == num_epochs:
            e = epoch + 1
            # print(f'Saving model at epoch {e}, accuracy {val_acc:.3f}')
            saveModel(RUN_TITLE, model)

    print('----- Finished Training -----')


# ----------------------------
# Inference
# ----------------------------
def inference(model, test_dl, device, model_dir, lrcn=False):
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
            inputs, labels, files = data[0].to(device), data[1].to(device), data[2]

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            if lrcn:
                inputs = (inputs, data[3])  # include lengths for LRCN model

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)

            # Count of predictions that are within 1 range the target label
            # correct_prediction += (prediction == labels).sum().item()
            correct_prediction += (abs(prediction - labels) <= 1).sum().item()
            total_prediction += prediction.shape[0]
            all_predictions.append(prediction)
            all_targets.append(labels)

            # append history to make into df later
            for i, clss in enumerate(labels):
                history.append([prediction[i].item(), clss.item(), files[i]])

        # make data frame of all predictions, targets, files
        df = pd.DataFrame(history, columns=['predictions', 'targets', 'file_stem'])
        test_acc = correct_prediction / total_prediction
        all_predictions = torch.cat(all_predictions).cpu()  # get long 1Dtensor of all preds
        all_targets = torch.cat(all_targets).cpu()
        altered_predictions = get_correct_list(all_targets, all_predictions)
        # f1_score(y_true, y_pred)
        f1 = f1_score(all_targets, altered_predictions, num_classes=5,
                      average='macro').item()  # f1_score returns a tensor, use item()

        # classification_report(y_true, y_pred)
        print(classification_report(all_targets, altered_predictions, labels=[0,1,2,3,4],
                                    target_names=['class 0', 'class 1', 'class 2', 'class 3', 'class 4']))

        # classification_report(y_true, y_pred)
        ConfusionMatrixDisplay.from_predictions(all_targets, altered_predictions, labels=[0, 1, 2, 3, 4], cmap=plt.cm.Blues)
        plt.savefig(os.path.join(model_dir, 'confusion_matrix_altered_predictions.png'))
        ConfusionMatrixDisplay.from_predictions(all_targets, all_predictions, labels=[0, 1, 2, 3, 4], cmap=plt.cm.Oranges)
        plt.savefig(os.path.join(model_dir, 'confusion_matrix_actual_predictions.png'))

        plot_roc_curve(y_test=all_targets, y_pred=altered_predictions, save_path=model_dir+'/ROC_curve.png')

        return test_acc, f1, total_prediction, df


def main(args):  # DEFAULTS
    DATA_PATH = args.data_directory  # ../recordings
    RUN_TITLE = args.run_title  # testing X
    # META_FILE = args.meta_data    ../recordings (to use if separate csv file)

    EPOCHS = args.epochs  # 100
    LEARNING_R = args.learningrate  # 0.1
    BATCH_SIZE = args.batchsize  # 64
    NORM_TYPE = args.norm_type  # global
    WEIGHTS = args.weights  # True
    MIX_PATIENTS = args.mix_patients  # False
    LRCN_ON = args.lrcn  # False
    INPUT_DIMS = args.input_dims  # 64
    HIDDEN_DIMS = args.hidden_dims  # 128

    model_dir = make_model_dir_path(RUN_TITLE)

    print(f'================================================================ \n '
          f'TRAINING {RUN_TITLE}'
          f'\n================================================================')


    if not MIX_PATIENTS:
        # get list of non-overlapping patients for train and test
        # train_patient_ids, test_patient_ids = ds.get_train_and_test_patient_ID_list(split_ratio=0.8, seed=SEED)
        # Creating data loaders
        # test_patient_ids = [12,13,29,31,34,4,41,5,8]  # same ids as 200ms
        # test_patient_ids = [16,17,23,26,33,41,5,6,8]
        # print('Using same ids as 500ms')  # same ids as 500ms
        test_patient_ids = [7, 10, 11, 27, 29, 34, 37, 38]  # same ids as seed 6
        print('Using same ids as seed 6')
        # test_patient_ids = [1, 6, 9, 10, 12, 13, 16, 38]  # final k-fold test
        # print('Using same ids as final k-fold test')
        train_patient_ids = [i for i in list(range(42))[1:] if i not in test_patient_ids]
        train_ds, train_dl, val_dl, test_dl = ds.make_data_loaders(DATA_PATH, BATCH_SIZE, NORM_TYPE, train_patient_ids, test_patient_ids, RUN_TITLE=RUN_TITLE, LRCN=LRCN_ON)
    else:  # speakers will potentially overlap in train and test
        train_ds, train_dl, val_dl, test_dl = ds.make_data_loaders_mixed_speakers(DATA_PATH, BATCH_SIZE, NORM_TYPE, RUN_TITLE=RUN_TITLE)
    size = train_ds[0][0].shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the model and put it on the GPU if available
    if LRCN_ON:
        myModel = LRCN(lstm_input_size=INPUT_DIMS, hidden_size=HIDDEN_DIMS)
        myModel.apply(weights_init)
        myModel = myModel.to(device)
        # for i in range(1):
        #     batch = next(iter(train_dl))
            # summary(myModel, input_size=(batch[0].shape, (64)))
    else:
        myModel = AudioClassifier(out_features=N_CLASSES)
        myModel.apply(weights_init)  # initialise parameters with normal distr.
        myModel = myModel.to(device)
        summary(myModel, input_size=(size[0], size[1], size[2]))  # C, H, W


    # Check that it is on Cuda
    print(f"Model is on: {next(myModel.parameters()).device}")

    print(f"Shape of sample: {size} = (seq_len), num_channels, freq_bins, num_sftf"
        f"Training model: epochs = {EPOCHS}, batch size = {BATCH_SIZE}, lr = {LEARNING_R}, norm_type = {NORM_TYPE}, class weights = {WEIGHTS}. Seed = {SEED}. LRCN = {LRCN_ON} (input_dims = {INPUT_DIMS}, hidden_dims = {HIDDEN_DIMS}) "
        f"{len(next(os.walk(DATA_PATH))[2])} audio files from {DATA_PATH}\n"
        f"================================================================")

    start_time_sec = time.time()
    training(myModel, train_dl, val_dl, EPOCHS, LEARNING_R, RUN_TITLE, device=device, weights=WEIGHTS, lrcn=LRCN_ON)
    end_time_sec = time.time()
    total_time_sec = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / EPOCHS
    print(f'Time total:     {total_time_sec / 60:5.2f} min')
    print(f'Time per epoch: {time_per_epoch_sec / 60:5.2f} min')

    write_training_history_to_file(history, RUN_TITLE)

    # plot and save graph
    if not args.graphoff:
        plot_results(history, EPOCHS, RUN_TITLE, model_dir)

    print(f'================================================================ \n '
          f'INFERENCE {RUN_TITLE}'
          f'\n================================================================')
    trained_model = load_model(RUN_TITLE, INPUT_DIMS, HIDDEN_DIMS, lrcn=LRCN_ON)
    print("----- Acc and F1 scores  -----")
    test_acc, f1, total_prediction, df = inference(trained_model, test_dl, device, model_dir, lrcn=LRCN_ON)
    mse, total_file, per_file_df = analyse_results(df)
    print(f'Test accuracy and macro f1 per frame: {test_acc:.2f} {f1:.2f}, Total test items: {total_prediction}')
    print(f'MSE for files: {mse}, Total num of files: {total_file}')

    # Save short summary of test run and of per_file results
    write_summmary(test_acc, total_prediction, mse, total_file, f1, RUN_TITLE)
    write_full_summary(per_file_df, RUN_TITLE, per_file=True)
    scatter_plot_per_patient(df, save_path=model_dir+'/patient_scatterplot.png')
    write_full_summary(df, RUN_TITLE, per_file=False)


if __name__ == '__main__':
    main(process_commandline())
