import os
import csv
import shutil
from pathlib import Path
from pandas import *

# # open and store the csv file
# audio_F1 = {}
# audio_F2 = {}
# with open('STX_VPD_metadata.csv','r') as csvfile:
#     metadata = csv.reader(csvfile, delimiter = ',')
#     # build dictionary with associated IDs
#     for row in metadata:
#         audio_F1[row[3]] = [str(row[0]), str(row[10])]
#         audio_F2[row[4]] = [str(row[0]), str(row[10])]
#         print(audio_F1[row[3]], audio_F2[row[4]])
#
# # move files
# D_list = [audio_F1, audio_F2]
# path = 'audio_83_copy'
#
# for file in os.listdir(path):
#     # ignore files in path which aren't in the csv file
#
#     file_stem = Path(file).stem
#     for D in D_list:
#         if file_stem in D:
#             newname = f'{D[file_stem][1]}_P{D[file_stem][0]}_{file_stem}.wav'
#             try:
#                 os.rename(os.path.join(path, file), os.path.join(path, newname))
#             except:
#                 print('File ' + oldname + ' could not be renamed to ' + D[file_stem])

path = 'extracted_audio'

data = read_csv('STX_VPD_metadata.csv')
patient = data['Patient'].tolist()
F1 = data['Video_File_1'].tolist()
F2 = data['Video_File_2'].tolist()
Class = data['Class'].tolist()

print(F1[0], type(F1[0]))
for file in os.listdir(path):
    file_stem = Path(file).stem
    if file_stem in F1:
        idx = F1.index(file_stem)
        newname = f'{int(Class[idx])}_{int(patient[idx])}_{file_stem}.wav'
        newpath = os.path.join('classed_audio', newname)
        if not os.path.exists(newpath):
            shutil.copy(os.path.join(path, file), newpath)
            print(f'Moved {file} to {newpath}')
    elif file_stem in F2:
        idx = F2.index(file_stem)
        newname = f'{int(Class[idx])}_{int(patient[idx])}_{file_stem}.wav'
        newpath = os.path.join('classed_audio', newname)
        if not os.path.exists(newpath):
            shutil.copy(os.path.join(path, file), newpath)
            print(f'Moved {file} to {newpath}')
