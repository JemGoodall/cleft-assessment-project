# ----------------------------
# Prepare the data - make a Dataset object from a directory of data
# Output: a torch dataset object
# to be imported in test.py and train.py

# adapt as necessary
# ----------------------------
import pandas as pd
import os
from torch.utils.data import Dataset
import torch
from helper_funcs import save_mean_and_std, get_mean_and_std
from torch.utils.data import DataLoader, Dataset, random_split
import random
from torchvision import transforms
from collate import PadSequence



# ----------------------------
# Get patient ID lists for train and test split
# ----------------------------
def get_train_and_test_patient_ID_list(split_ratio=0.8, seed=4):
    random.seed(seed)
    total_patients = list(range(42))[1:]  # MUST BE HARD CODED!!
    num_train_samples = round(split_ratio * len(total_patients))

    # get random list of patients for train and test
    train_patient_ids = random.sample(range(0, len(total_patients)), num_train_samples)

    test_patient_ids = [i for i in total_patients if i not in train_patient_ids]

    return train_patient_ids, test_patient_ids


# Takes a directory of data and returns a dataframe with the files from the patients specified in the patient_id_list
def get_df(data_dir, patient_id_list=None):
    data = []
    i = 1
    for filename in os.listdir(data_dir):
        patient_id = int(filename.split('_')[1])  # take the second val in the file name which indicates the patient
        if patient_id_list:
            if patient_id in patient_id_list:
                class_name = int(filename[0])
                data.append([filename, class_name])
                i +=1
        else:  # make the dataframe with all the patients in the data_dir
            class_name = int(filename[0])
            data.append([filename, class_name])
    df = pd.DataFrame(data, columns=['relative_path', 'ClassID'])
    # print(df.head())

    return df


# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, df, data_path, transform=None, LRCN=False):
        self.df = df  # df determines which speakers get included: columns=['relative_path', 'ClassID']
        self.df["original_file"] = self.df["relative_path"].replace(r'_([\d])+\.pt', '', regex=True)  # new column to get original file name
        self.data_path = str(data_path)
        self.transform = transform
        self.LRCN = LRCN  # need df with only original file names (no chunks)
        if self.LRCN:
            self.df_LRCN = self.df[['ClassID', 'original_file']].copy().drop_duplicates().reset_index(drop=True)
            print(f'df_LRCN length = {len(self.df_LRCN)}')
            # print(f'counts of classes: {list(self.df_LRCN["ClassID"].value_counts(normalize=True))}')
    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        if self.LRCN:
            return len(self.df_LRCN)
        else:
            return len(self.df)

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        if self.LRCN:
            # get the original file name and loop through the chunks
            original_file = self.df_LRCN.loc[idx, 'original_file']
            chunk_list = []
            for chunk_path in self.df.groupby('original_file').get_group(original_file)['relative_path']:
                chunk_file = os.path.join(self.data_path, chunk_path)
                sgram = torch.load(chunk_file)
                if self.transform:
                    sgram = self.transform(sgram)
                chunk_list.append(sgram)
            sgram_seq = torch.stack(chunk_list)

            # sgram_seq.shape [T, C(1), mel_bins(64), stft(38)]

            class_id = self.df_LRCN.loc[idx, 'ClassID']  # Get the Class ID

            return sgram_seq, class_id, original_file

        else:
            # Absolute file path of the audio file - concatenate the audio directory with the relative path
            relative_path = self.df.loc[idx, 'relative_path']
            sgram_file = os.path.join(self.data_path, relative_path)
            # open and load the tensor for a single sgram
            sgram = torch.load(sgram_file)
            if self.transform:
                sgram = self.transform(sgram)

            class_id = self.df.loc[idx, 'ClassID']  # Get the Class ID

            return sgram, class_id, relative_path


def get_random_split(dataset, split=0.8):
    # Random split of 80:20 between training and test
    num_items = len(dataset)
    num_train = round(num_items * split)
    num_val = num_items - num_train
    train_ds, test_ds = random_split(dataset, [num_train, num_val], generator=torch.Generator().manual_seed(42))

    return train_ds, test_ds


def make_data_loaders(DATA_PATH, BATCH_SIZE, NORM_TYPE, train_patient_ids, test_patient_ids, RUN_TITLE=None, LRCN=False):
    train_df = get_df(DATA_PATH, patient_id_list=train_patient_ids)
    test_df = get_df(DATA_PATH, patient_id_list=test_patient_ids)
    print(f'Train patients = {len(train_patient_ids)}, Test patients = {len(test_patient_ids)}')
    print(f'Train: {train_patient_ids}, \nTest = {test_patient_ids}')
    print(f'Num train samples = {train_df.shape[0]}, Num test samples = {test_df.shape[0]}')
    if NORM_TYPE == "off":
        myds_train = SoundDS(train_df, DATA_PATH)
        myds_test = SoundDS(test_df, DATA_PATH)
    else:
        # # Process audio into spectrogram dataset
        myds_ = SoundDS(train_df, DATA_PATH)
        # get mean and std for normalisation
        mean, std = get_mean_and_std(myds_, norm_type=NORM_TYPE)
        # save scaler values
        if RUN_TITLE:
            save_mean_and_std(mean, std, RUN_TITLE)

        # apply normalisation - Normalize works PER CHANNEL
        myds_train = SoundDS(train_df, DATA_PATH, transform=transforms.Normalize(mean, std), LRCN=LRCN)
        myds_test = SoundDS(test_df, DATA_PATH, transform=transforms.Normalize(mean, std), LRCN=LRCN)
        # on each sample, where tensor shape [channel, H, W]

    # Random split of 80:20 between training and validation
    train_ds, val_ds = get_random_split(myds_train)

    # Create training and validation data loaders
    if LRCN:
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=PadSequence(), generator=torch.Generator().manual_seed(0))
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=PadSequence())
        test_dl = torch.utils.data.DataLoader(myds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=PadSequence())

        return train_ds, train_dl, val_dl, test_dl

    else:
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(0))
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_dl = torch.utils.data.DataLoader(myds_test, batch_size=BATCH_SIZE, shuffle=False)

        return train_ds, train_dl, val_dl, test_dl


def make_data_loaders_mixed_speakers(DATA_PATH, BATCH_SIZE, NORM_TYPE, RUN_TITLE=None):
    print('WARNING: mix_patients=True. Train and test datasets may contain files from the same speaker.')
    df = get_df(DATA_PATH)
    if NORM_TYPE == "off":
        myds = SoundDS(df, DATA_PATH)
    else:
        # # Process audio into spectrogram dataset
        myds_ = SoundDS(df, DATA_PATH)
        # get mean and std for normalisation
        mean, std = get_mean_and_std(myds_, norm_type=NORM_TYPE)
        # save scaler values
        if RUN_TITLE:
            save_mean_and_std(mean, std, RUN_TITLE)

        # apply normalisation - Normalize works PER CHANNEL
        myds = SoundDS(df, DATA_PATH, transform=transforms.Normalize(mean, std))

    train_ds_whole, test_ds = get_random_split(myds)
    train_ds, val_ds = get_random_split(train_ds_whole)

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_ds, train_dl, val_dl, test_dl

