'''
For making a spectrogram of a .wav file.
Using the same methods as in pre_process.py
(the same methods used for the CNN)
Exception: no padding for better visualisation

use: make_spec_images.py [audio file path] [title of plot]
output: a plot in the cwd, saved as [audio file path].png
'''

from pre_process import AudioUtil, SoundDS
import matplotlib.pyplot as plt
import torchaudio
import librosa
import sys
import os

try:
    audio_file = sys.argv[1]
except IndexError:
    print("Please provide a file path to .wav")
    sys.exit()

try:
    title = sys.argv[2]
except IndexError:
    print("Please provide a title for the plot")
    sys.exit()

# audio_file = '../recordings/0_jackson_8.wav'
# title = 'spec'

SR = 8000
CHANNEL = 2
DUR = 4000
SHIFT_PCT = 0.0
N_MELS = 128
N_FFT = 1024

# -------
# Pipe line taken from pre_process.py (excluding padding)
# -------
def make_spec(audio_file):
    aud = AudioUtil.open(audio_file)
    reaud = AudioUtil.resample(aud, SR)
    rechan = AudioUtil.rechannel(reaud, CHANNEL)
    # new_aud = AudioUtil.pad_trunc(rechan, DUR)
    new_aud = AudioUtil.time_shift(rechan, SHIFT_PCT)
    sgram = AudioUtil.spectro_gram(new_aud, n_mels=N_MELS, n_fft=N_FFT)

    return sgram

def plot_save_spectrogram(specgram, save_file, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(specgram, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.savefig(f'{save_file}.png')


path, filename = os.path.split(os.path.abspath(audio_file))
save_file = filename.split('.')[0]
sgram = make_spec(audio_file)
plot_save_spectrogram(sgram[0], save_file, title=title)  # only take first channel

