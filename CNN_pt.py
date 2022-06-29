'''https://stackoverflow.com/questions/59584457/pytorch-is-there-a-definitive-training-loop-similar-to-keras-fit'''
import pre_process as PreP
import meta_data
import sys
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import csv
import os
import pandas as pd
import time

'''HYPER-PARAMETERS '''
N_CLASSES = 10
EPOCHS = 10
LEARNING_R = 0.001
BATCH_SIZE = 16
SHIFT = False
MASK = False

try:
    DATA_PATH = sys.argv[1]
except IndexError as e:
    print(e, ": Please provide a path to the dataset")
    sys.exit()
try:
    run_title = sys.argv[3]  # change this with actual data to 2
except IndexError:
    print("Please provide a test run title")
    sys.exit()
log_file = 'log.csv'
df = meta_data.get_df(DATA_PATH)
# df = meta_data.get_df('../recordings')
print(df.head())

# Process audio into spectrogram dataset
myds = PreP.SoundDS(df, DATA_PATH, shift=SHIFT, mask=MASK)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Each batch has a shape of (batch_sz, num_channels, Mel freq_bands, time_steps) E.G. torch.Size([16, 2, 64, 344])

# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=N_CLASSES)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x


# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
print(f"Model is on: {next(myModel.parameters()).device}")

history = {}  # Collects per-epoch loss and acc like Keras' fit().
history['epoch'] = []
history['loss'] = []
history['acc'] = []
history['val_loss'] = []
history['val_acc'] = []


# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, val_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
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
        val_loss = 0.0
        val_correct_prediction = 0
        val_total_prediction = 0


        for i, data in enumerate(val_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # forward
            val_outputs = model(inputs)
            loss = criterion(val_outputs, labels)

            # Keep stats for Loss and Accuracy
            val_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(val_outputs, 1)
            # Count of predictions that matched the target label
            val_correct_prediction += (prediction == labels).sum().item()
            val_total_prediction += prediction.shape[0]

        # Print stats at the end of the epoch
        num_batches = len(val_dl)
        val_loss = val_loss / num_batches
        val_acc = val_correct_prediction / val_total_prediction

        print(f'Epoch: {epoch+1}/{num_epochs} | Loss: {avg_loss:.2f}, Accuracy: {acc:.2f} |'
              f' Val_loss: {val_loss:.2f}, Val_ accuracy: {val_acc:.2f}')
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch'].append(epoch+1)


    print('Finished Training')


# ----------------------------
# Inference
# ----------------------------
def inference(model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

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

    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')


def main():
    print(f"shape of sample: {train_ds[2][0].shape}")
    print(f"Training model: epochs = {EPOCHS}, batch size = {BATCH_SIZE}. {len(train_ds)} audio files from {DATA_PATH}")
    start_time_sec = time.time()
    training(myModel, train_dl, val_dl, EPOCHS)
    end_time_sec = time.time()
    total_time_sec = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / EPOCHS
    print(f'Time total:     {total_time_sec/60:5.2f} min')
    print(f'Time per epoch: {time_per_epoch_sec/60:5.2f} min')
    with open('log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([run_title])
    pd.DataFrame(history).to_csv('log.csv', index=False, mode='a', sep='|')
    print("----- Inference step -----")
    # inference(myModel, test_dl)


if __name__ == '__main__':
    main()
