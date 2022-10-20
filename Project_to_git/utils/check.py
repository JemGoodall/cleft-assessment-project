'''
For hyper parameter tuning
'''
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import make_dataset as ds
import sys
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics.functional import f1_score
from helper_funcs import get_correct_list, analyse_results
import pandas as pd
from numpy import array
from sklearn.model_selection import KFold
from model import AudioClassifier, weights_init, LRCN

torch.manual_seed(0)  # reproducibility!
torch.set_printoptions(linewidth=120)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# set up global things
# RUN_DIR = 'hp_tuning_trial3'
DATA_DIR = sys.argv[1]
# TRAIN_DIR = os.path.join(DATA_DIR, 'test_data')  # assumes the data has already been split into train and test dirs
# TEST_DIR = os.path.join(DATA_DIR, 'test_data')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_CLASSES = 5
NORM_TYPE = 'global'
K = 5
LRCN_ON = True
weights = [1.7868, 3.8227, 1.5828, 1.138, 1.0]  # balanced dataset
class_weights = torch.FloatTensor(weights).cuda()

# --> Hyperparams
parameters = dict(
    test=[64, 128],
    test2=[128, 256, 512]
)

# params set here just for quick checks
lr = 0.001
num_epochs = 100
batch_size = 64
# kernel_size = (3, 3)
# max_pool = (2, 2)
# stride = (2, 2)

param_values = [v for v in parameters.values()]


##### BUILD CNN MODEL ##### ==============================================

class audioClassifier(nn.Module):
    def __init__(self, kernel_size=(3, 3), max_pool=(2, 2), stride=(2, 2)):
        super().__init__()
        # First Convolution Block with Relu and Batch Norm.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(32)

        # Second Convolution Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(64)

        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()

        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=max_pool, stride=max_pool,
                                 padding=(1, 1))  # determine this later! padding=1 because
        # kernel size probably doesn't fit the input evenly

        # Define proportion of nodes to dropout
        self.dropout = nn.Dropout(p=0.1)

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=N_CLASSES)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output (don't softmax, taken care of in nn.CrossEntropyLoss()
        return x


# def weights_init(m):  # initialise parameters with normal distribution
#     if isinstance(m, nn.Conv2d) or isinstance(m, torch.nn.Linear):
#         torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.1)
#         torch.nn.init.normal_(m.bias.data, mean=0.0, std=0.1)


##### MAKE DATA SET SPLITS ##### ==============================================

k_list = []
total_patients = array(list(range(42))[1:])
kfold = KFold(n_splits=K, shuffle=True, random_state=1)
for train, test in kfold.split(total_patients):
    k_list.append([list(total_patients[train]), list(total_patients[test])])

# k_list = [[[2, 3, 4, 5, 7, 8, 11, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41],
# [1, 6, 9, 10, 12, 13, 16, 38]]]
# print(k_list)


# print(f'len of klist = {len(k_list)}, len_train={len(k_list[0][0])}, len_test={len(k_list[0][1])}')
for idx in k_list[0][0]:
    if idx in k_list[0][1]:
        print('ALERT!! TEST AND TRAIN HAVE SAME SPEAKERS!')
sample_shape = torch.load(os.path.join(DATA_DIR, os.listdir(DATA_DIR)[0])).shape

info = f"Shape of each sample: {sample_shape}, Norm type = {NORM_TYPE} Data source: {DATA_DIR}, Weights={weights}, K-folds = {K}, lrcn = {LRCN_ON}"

print('=========================================================')
print(f'k-fold train and test run \nHParameters = {parameters}\n{info}')
print('=========================================================')

##### TRAINING LOOP ##### ==============================================

for run_id, (inputD, hiddenD) in enumerate(product(*param_values)):
    name = f"TLRCN{inputD}H{hiddenD}"
    print(f'################## Starting Run: {run_id + 1} on {name} ##################')

    if LRCN_ON:
        model = LRCN(lstm_input_size=inputD, hidden_size=hiddenD).apply(weights_init).to(DEVICE)
    else:
        model = AudioClassifier().apply(weights_init).to(DEVICE)

    # tb = SummaryWriter(os.path.join('experiments', name))
    tb = SummaryWriter(comment=name)  # set up tensorboard for individual run
    tb.add_text(tag='info', text_string=f'{name} \n{info}', global_step=0)

    # cross-validation values
    k_loss = 0.0
    k_acc = 0.0
    k_vloss = 0.0
    k_vacc = 0.0
    k_testacc = 0.0
    k_testf1 = 0.0
    k_mae = 0.0

    k_run = 0
    for data_set_ids in k_list:  # where k_list is [[train_ids, test_ids], [train_ids, test_ids]...]
        k_run += 1
        train_patient_ids, test_patient_ids = data_set_ids

        _, train_dl, val_dl, test_dl = ds.make_data_loaders(DATA_DIR, batch_size, NORM_TYPE,
                                                            train_patient_ids, test_patient_ids, LRCN=LRCN_ON)

        criterion = nn.CrossEntropyLoss(weight=class_weights)  # applies nn.LogSoftmax() and nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr / 10,
                                                        steps_per_epoch=int(len(train_dl)),
                                                        epochs=num_epochs,
                                                        anneal_strategy='linear')

        for epoch in range(num_epochs):

            # --- TRAIN ON TRAINING SET -------------------------------------
            # Repeat for each batch in the training set
            model.train()
            running_loss = 0.0
            correct_prediction = 0
            total_prediction = 0

            for i, data in enumerate(train_dl):
                # Get the input features and target labels, and put them on the GPU
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                if LRCN_ON:
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
                    inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                    # forward
                    if LRCN_ON:
                        inputs = (inputs, data[3])  # include lengths for LRCN model
                    val_outputs = model(inputs)  # outputs torch.Size([batchsize, classes])
                    loss = criterion(val_outputs, labels)

                    # Keep stats for Loss and Accuracy
                    val_running_loss += loss.item()

                    # Get the predicted class with the highest score
                    _, prediction = torch.max(val_outputs, 1)  # max over classes (dim=1)
                    # Count of predictions that matched the target label
                    val_correct_prediction += (prediction == labels).sum().item()
                    val_total_prediction += prediction.shape[0]

            # Print stats at the end of the epoch
            num_batches = len(val_dl)
            val_loss = val_running_loss / num_batches
            val_acc = val_correct_prediction / val_total_prediction

            # print(f'Epoch: {epoch + 1}/{num_epochs} | Loss: {avg_loss:.3f}, Accuracy: {acc:.3f} |'
            #       f' Val_loss: {val_loss:.3f}, Val_accuracy: {val_acc:.3f}')

            if k_run == K:  # only add scalar values for final run to keep it tidy
                tb.add_scalars("Loss", {'train': avg_loss, 'val': val_loss}, epoch + 1)
                tb.add_scalars("Accuracy", {'train': acc, 'val': val_acc}, epoch + 1)

            if epoch + 1 == num_epochs:  # add final epoch values
                print(f'FINAL EPOCH {epoch + 1}/{num_epochs}: ACCUMULATING K VALUES...')
                k_loss += avg_loss
                k_acc += acc
                k_vloss += val_loss
                k_vacc += val_acc

        print(f'------------------ Finished {k_run}/{K} Training for: {name} ------------------')

        model.eval()
        cp = 0
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

                # # Normalize the inputs
                # inputs_m, inputs_s = inputs.mean(), inputs.std()
                # inputs = (inputs - inputs_m) / inputs_s

                # Get predictions
                if LRCN_ON:
                    inputs = (inputs, data[3])  # include lengths for LRCN model
                outputs = model(inputs)

                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs, 1)

                # Count of predictions that are within 1 range the target label
                cp += (prediction == labels).sum().item()
                correct_prediction += (abs(prediction - labels) <= 1).sum().item()
                total_prediction += prediction.shape[0]
                all_predictions.append(prediction)
                all_targets.append(labels)

                # append history to make into df later
                for i, clss in enumerate(labels):
                    history.append([prediction[i].item(), clss.item(), files[i]])

            # make data frame of all predictions, targets, files
            df = pd.DataFrame(history, columns=['predictions', 'targets', 'file_stem'])
            mae, _, _ = analyse_results(df)

            test_acc = correct_prediction / total_prediction
            ta = cp / total_prediction

            all_predictions = torch.cat(all_predictions).cpu()  # get long 1Dtensor of all preds
            all_targets = torch.cat(all_targets).cpu()
            altered_predictions = get_correct_list(all_targets, all_predictions)
            f1 = f1_score(all_targets, altered_predictions, num_classes=5,
                          average='macro').item()  # f1_score returns a tensor, use item()
            f1_true = f1_score(all_targets, all_predictions, num_classes=5,
                               average='macro').item()

            print(f'Unaltered results: test accuracy = {ta:.2f}, macro f1 = {f1_true}')

            # classification_report(y_true, y_pred)
            print(classification_report(all_targets, altered_predictions, labels=[0, 1, 2, 3, 4],
                                        target_names=['class 0', 'class 1', 'class 2', 'class 3', 'class 4'],zero_division=0))

            k_testacc += test_acc
            k_testf1 += f1
            k_mae += mae
        print(
            f'--------- {k_run}/{K} Test results for: {name}, acc={test_acc} f1={f1} mae={mae} ---------')

    print(f'FINAL accumulated results for {name}: k_acc={k_testacc} k_f1={k_testf1} k_mae={k_mae}, (kloss,kacc){k_vloss, k_vacc}')

    tb.add_hparams(
        {"Input": inputD, "Hidden": hiddenD},
        {
            "cross/train_accuracy": k_acc / K,
            "cross/val_accuracy": k_vacc / K,
            "cross/train_loss": k_loss / K,
            "cross/val_loss": k_vloss / K,
            "cross/test_accuracy": k_testacc / K,
            "cross/test_f1_score": k_testf1 / K,
            "cross/mae": k_mae / K,

        },
        run_name='HyperParams'
    )

    d = {"Input": inputD, "Hidden": hiddenD,
         "cross/train_accuracy": k_acc / K,
         "cross/val_accuracy": k_vacc / K,
         "cross/train_loss": k_loss / K,
         "cross/val_loss": k_vloss / K,
         "cross/test_accuracy": k_testacc / K,
         "cross/test_f1_score": k_testf1 / K,
         "cross/mae": k_mae / K,
         }
    print(d)
    tb.close()


