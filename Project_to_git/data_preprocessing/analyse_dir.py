import librosa
import torchaudio
import sys
import os

INPUT_DIR = sys.argv[1]
num_files = 0
dur = 0
dur_set = []
sr = 0
speaker_list = []
# speaker_list_mono = [3,6,8,15,19,21,22,23,24,25,27,28,29,30,32,34,36,41,1,2,9,17,35]
files_list = []
all_classes = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
for filename in os.listdir(INPUT_DIR):
    file_path = os.path.join(INPUT_DIR, filename)
    classification, speaker_id, file_id, _ = filename.split('_')
    speaker_list.append(speaker_id)
    files_list.append(file_id)
    all_classes[int(classification)] += 1
    num_files += 1
    _dur = librosa.get_duration(filename=file_path)
    sig, _sr = torchaudio.load(file_path)
    print(f'calculating file {filename}, dur={_dur}, speaker_id={speaker_id}')
    dur += _dur
    dur_set.append(_dur)
    sr += _sr

print(f'num of files {num_files}')
print(f'avg duration in sec = {dur / num_files}')
print(f'avg sr = {sr / num_files}')
print(f'num unique speakers = {len(set(speaker_list))}')
print(f'num unique audio files = {len(set(files_list))}')
# print((set(files_list)))
print(f'num of class types = {all_classes}')
