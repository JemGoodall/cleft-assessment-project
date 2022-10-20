'''
For making a spectrogram of a .wav file.
Using the same methods as in pre_process.py
(the same methods used for the CNN)
Exception: no padding for better visualisation

use: make_spec_images.py [audio file path] [title of plot]
output: a plot in the cwd, saved as [audio file path].png
'''
import sys
sys.path.insert(0, '/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1509050_Jemima_Goodall/thesis/v3_code')
from make_spec_data import AudioUtil
import matplotlib.pyplot as plt
import torch
import librosa
import os
import librosa.display
import numpy as np
# try:
#     audio_file = sys.argv[1]
# except IndexError:
#     print("Please provide a file path to .wav")
#     sys.exit()
#
# try:
#     title = sys.argv[2]
# except IndexError:
#     print("Please provide a title for the plot")
#     sys.exit()

# -------
# Taken from make_spec_data.py
# -------
def make_spec(audio_file, type):
    aud = AudioUtil.open(audio_file)
    # new_aud = AudioUtil.pad_trunc(aud, DUR)  # padding
    sgram = AudioUtil.spectro_gram(aud, spec_type=type, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)

    # elif type == 'cqt':
    #     sig = new_aud[0].numpy()
    #     C = np.abs(librosa.cqt(sig, sr=SR, n_bins=N_FFT//2 + 1, hop_length=N_FFT//2))
    #     sgram = librosa.amplitude_to_db(C, ref=np.max)
    #     sgram = torch.from_numpy(sgram)

    return sgram


def plot_save_spectrogram(specgram, save_file, graph_type, title=None):
    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(1, 1)

    if graph_type == 'frames':
        # axs.set_title(title or "Spectrogram (db)")
        axs.set_ylabel("Mel-scaled Frequency Bins")
        axs.set_xlabel("STFTs")
        im = axs.imshow(specgram, origin="lower", aspect="auto")
        # fig.colorbar(im, ax=axs)

    elif graph_type == 'time':
        img = librosa.display.specshow(specgram.numpy(), sr=sr, x_axis='Time(s)', y_axis='Frequency Bins', ax=axs)
        # axs.set_title(title)
        fig.colorbar(img, ax=axs, format="%+2.0f dB")

    # plt.show()
    plt.savefig(f'../final_images/{save_file}.png',bbox_inches='tight')



# SR = 22050
SR = 22050
CHANNEL = 1
DUR = 200
SHIFT_PCT = 0.0
N_MELS = 128
N_FFT = 512
HOP = N_FFT//2
file = '0_1_389723_3_2'
audio_file = f'../processed_data/UTTS_200ms_nopad/{file}.wav'
# audio_file = f'../CHILD_DATA_perUtt/{file}.wav'
title = f'mels{N_MELS}_f{N_FFT}_h{HOP}'
type = 'Mel'
graph_type = 'frames'
path, filename = os.path.split(os.path.abspath(audio_file))
dur = librosa.get_duration(filename=audio_file)
sig, sr = AudioUtil.open(audio_file)
print(f'duration in sec = {dur}')
print(f'number of original samples = {sig.shape[1]}, original sr = {sr}')
sgram_mels = make_spec(audio_file, type=type)
print(sgram_mels.type(), sgram_mels.dtype, sgram_mels.shape)
save_file = title
# aud = AudioUtil.open(audio_file)
# sig, sr = AudioUtil.resample(aud, SR)
# print(f'number of new samples = {sig.shape[1]}, new sr = {sr}')
# print(sgram[0].shape)


# sgram_cqt = make_spec(audio_file, type='cqt')
#
# print(sgram_cqt.type(), sgram_cqt.dtype, sgram_cqt.shape)
plot_save_spectrogram(sgram_mels[0], save_file, graph_type=graph_type, title=title)  # only take first channel

