import librosa
import torchaudio
import sys

audio_file = sys.argv[1]

dur = librosa.get_duration(filename=audio_file)
sig, sr = torchaudio.load(audio_file)
print(f'duration in sec = {dur}')
print(f'number of original samples = {sig.shape[1]}, original sr = {sr}, num of channels = {sig.shape[0]}')
