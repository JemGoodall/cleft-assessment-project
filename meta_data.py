# ----------------------------
# Prepare training data from Metadata file
# output: dataframe, with the relative path of each audio
# and the corresponding class

# adapt as necessary
# ----------------------------
import pandas as pd
from pathlib import Path
import sys
import os

# download_path = Path.cwd()/'UrbanSound8K'
#
# # Read metadata file
# metadata_file = download_path/'metadata'/'UrbanSound8K.csv'


# try:
#     metadata_file = sys.argv[1]
# except IndexError as e:
#     print(e, ': Please provide a [TEMP] filepath ...')
#     sys.exit()
#

def get_df(metadata_file):
    ''''FOR TESTING ON DIGIT DATA'''
    if sys.argv[2] == 'test':
        data = []
        for filename in os.listdir(metadata_file):
            if "wav" in filename:
                class_name = int(filename[0])
                data.append([filename, class_name])

        df = pd.DataFrame(data, columns=['relative_path', 'ClassID'])

         # print(df.head())
        return df

    else:
        '''IF CVS FILE'''
        df = pd.read_csv(metadata_file)
        df.head()

        # Construct file path by concatenating fold and file name
        df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

        # Take relevant columns
        df = df[['relative_path', 'classID']]  # !!!to do: make sure th classID is an INT.
        # print(df.head())
        return df