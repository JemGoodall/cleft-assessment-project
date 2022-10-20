"""
FOR TESTING HNR FUNCTION ONLY
takes in a dir of audio files
returns a dir of preprocessed audio files
python HNR_test.py --input_dir --output_dir HNR_val

preprocessing steps:
- split audio in x ms chunks
- remove unvoiced chunks (HNR)
- put chunks back together
- save to new dir

"""
import scipy.io.wavfile
import os
from pathlib import Path
import sys
import numpy as np
from math import ceil
import argparse
import Signal_Analysis.features.signal
from Signal_Analysis import features

class ProcessAudio():
    def __init__(self, filepath):
        self.filepath = filepath
        self.rate, self.aud = scipy.io.wavfile.read(self.filepath)
        self.dur = self.aud.shape[0]/self.rate

    # def get_dur_in_sec(self):
    #     num_samples = self.aud.shape[0]
    #     dur = num_samples/self.rate
    #     return dur

    def get_aud(self):
        return self.aud


    @staticmethod
    def split_audio(aud, chunk_length_s=0.2, sample_rate_hz=8e3):
        """ Accepts a numpy array of audio data and splits it into chunks of equal length
            :param aud: numpy array of audio data
            :param chunk_length: length of chunk in seconds
            :param sample_rate_hz: the sample rate of the aud
            :return: list of audio chunks
            """
        dur_in_s = len(aud)/sample_rate_hz
        number_of_chunks = ceil(dur_in_s) / float(chunk_length_s)
        samples_per_chunk = int(chunk_length_s * sample_rate_hz)
        chunk_list = [aud[i * samples_per_chunk:(i + 1) * samples_per_chunk]
                for i in range(int(number_of_chunks))]
        return chunk_list

    @staticmethod
    def remove_unvoiced(aud_list, HNR_threshold = 50, sr=8e3):
        """ Accepts a list of audio arrays and removes one that are 'unvoiced' according to the HNR threshold
                    :param aud_list: list of numpy array audio data
                    :param HNR_threshold = the threshold for deciding if a segment is voiced or not
                    :return: list of audio chunks (only voiced chunks)
                    """
        hnr_list = []
        chunks = []
        for i, chunk in enumerate(aud_list):
            if len(chunk) == 0:
                print('chunk was empty!')
            else:
                chunk_hnr = features.signal.get_HNR(chunk, sr, silence_threshold=0.01)
                hnr_list.append(chunk_hnr)
                chunks.append(chunk)
        threshold = np.percentile(hnr_list, HNR_threshold)
        voiced_chunks = [ch for ch, hnr_val in zip(chunks, hnr_list) if hnr_val > threshold]
        print(f'LEN of LIST OF HNR: {len(hnr_list)} \n'
              f'THRESHOLD = {threshold}, '
              f'LEN of NEW LIST = {len(voiced_chunks)}')
        print(f'   {len(aud_list)- len(voiced_chunks)} chunks dropped, not enough voicing.')
        voiced_aud = np.concatenate(voiced_chunks, axis=0)
        return voiced_aud


def process_and_save(input_dir, output_dir, HNR_val, c_l=0.5):
    count = 1
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        print(f'opening {file_path}')
        audio_obj = ProcessAudio(file_path)
        aud = audio_obj.get_aud()
        print('---making chunks---')
        splits = ProcessAudio.split_audio(aud, chunk_length_s=c_l,sample_rate_hz=audio_obj.rate)
        print('---removing unvoiced chunks---')
        voiced_aud = ProcessAudio.remove_unvoiced(splits, HNR_threshold=HNR_val, sr=audio_obj.rate)
        # console output
        input_len = len([name for name in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, name))])
        print(f'PROCESSED [{count}/{input_len}] : {file_path}')
        count += 1

        # save new audio file
        file_stem = Path(file_path).stem
        new_file_name = f'{file_stem}_HNR{HNR_val}.wav'
        dir_path = os.path.join(output_dir, new_file_name)
        # print(f'Saving to {dir_path}')
        scipy.io.wavfile.write(dir_path, audio_obj.rate, voiced_aud)

    output_len = len([name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name))])
    print(f'Completed preprocessing: {input_len} files now converted to {output_len} in {output_dir}')


def process_commandline():
    parser = argparse.ArgumentParser(
        description='To preprocess audio data')

    # basic parameter arguments
    parser.add_argument('input_dir', default="../cleft_recordings",
                        help="Directory containing raw audio data")
    parser.add_argument('output_dir', default="../processed_data/audio",
                        help="Path to save processed data")
    parser.add_argument('HNR_val', default=50,
                        help="HNR threshold")
    args = parser.parse_args()

    # Must have data directory and run title
    if not args.input_dir or not args.output_dir:
        parser.error('Must supply, input and output dir.')

    return args


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    HNR_val = int(args.HNR_val)
    try:
        os.makedirs(output_dir)
    except FileExistsError as e:
        print(e)
        answer = input("Do you want to continue? [y/n]:")
        if answer == 'n':
            sys.exit()

    process_and_save(input_dir, output_dir, HNR_val, c_l=0.2)

if __name__ == '__main__':
    main(process_commandline())