"""
takes in a dir of audio files
returns a dir of preprocessed/split audio files with extended file names

python preprocess.py --input_dir --output_dir

preprocessing steps:
- remove noise(?)
- split audio in x ms chunks
- remove unvoiced chunks (HNR)
- save to new dir with new filename

"""
import scipy.io.wavfile
import os
from pathlib import Path
import sys
import numpy as np
from math import ceil, floor
import argparse
import Signal_Analysis.features.signal
from Signal_Analysis import features


class ProcessAudio():
    def __init__(self, filepath):
        self.filepath = filepath
        self.rate, self.aud = scipy.io.wavfile.read(self.filepath)
        self.dur = self.aud.shape[0] / self.rate

    def get_aud(self):
        return self.aud


    @staticmethod
    def remove_silence(aud, min_silence_duration=0.10, noise_threshold=150, sample_rate_hz=8e3):
        """ Accepts a numpy array of audio data and splits it at the points of silence into multiple arrays of data.
            And combines them back into one array.
        :param aud: numpy array of audio data
        :param min_silence_duration: the required period of silence to split the recording
        :param sample_rate_hz: the sample rate of the audio
        :return: numpy array of audio data without silence
        """
        # A list of tuples (start, stop)
        print(f'noise threshold = {noise_threshold}, min_silence_duration = {min_silence_duration}')
        min_silence_frame = sample_rate_hz * min_silence_duration
        silence_zones = []

        zone_start = None
        zone_end = None

        for idx, point in enumerate(aud):
            if abs(point) < noise_threshold and zone_start is None:
                zone_start = idx

            if abs(point) > noise_threshold and zone_start is not None:
                zone_end = idx

            # If we are in a silent zone and we come to the end point
            if zone_start is not None and zone_end and abs(point) > noise_threshold:
                if (zone_end - zone_start) > min_silence_frame:
                    silence_zones.append((zone_start, zone_end))

                zone_start = None
                zone_end = None

        # Split the recording by the zones and convert to single np array
        split_recordings = []
        for idx, zone in enumerate(silence_zones):
            if idx == 0:
                start = 0
            else:
                start = silence_zones[idx - 1][1]

            end = zone[0]
            split_recordings.append(aud[start:end])
        try:
            new_aud = np.concatenate(split_recordings, axis=0)
        except ValueError as e:
            print(f"{e}: No silence found. Consider increasing noise_threshold (={noise_threshold}) "
                  f"or decreasing silence_dur (={min_silence_duration})")
            return aud
        else:
            print(
                f'   Removing silence... Reducing {len(aud) / sample_rate_hz:.2f}s to {len(new_aud) / sample_rate_hz:.2f}s')
            return new_aud

    @staticmethod
    def split_audio(aud, chunk_length_s=0.2, sample_rate_hz=22050):
        """ Accepts a numpy array of audio data and splits it into chunks of equal length
            :param aud: numpy array of audio data
            :param chunk_length: length of chunk in seconds
            :param sample_rate_hz: the sample rate of the aud
            :return: list of audio chunks
            """
        dur_in_s = len(aud) / sample_rate_hz
        print(f'dur_in_s {dur_in_s}')
        number_of_chunks = floor(dur_in_s / float(chunk_length_s))
        print(f'num_chunks = {number_of_chunks}')
        samples_per_chunk = int(chunk_length_s * sample_rate_hz)
        print(f'samples per chunk = {samples_per_chunk}')
        chunk_list = [aud[i * samples_per_chunk:(i + 1) * samples_per_chunk]
                      for i in range(int(number_of_chunks))]
        return chunk_list

    @staticmethod
    def remove_unvoiced(aud_list, HNR_threshold=50, sr=8e3):
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
              f'THRESHOLD = {threshold} \n'
              f'LEN of NEW LIST = {len(voiced_chunks)}')
        print(f'   {len(aud_list) - len(voiced_chunks)} chunks dropped, not enough voicing.')
        return voiced_chunks


def process_and_save(input_dir, output_dir, c_l=0.5, HNR=50):
    count = 1
    input_len = len([name for name in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, name))])
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        print(f'opening {file_path}')
        audio_obj = ProcessAudio(file_path)
        aud = audio_obj.get_aud()
        # no_noise = ProcessAudio.remove_noise(aud)
        # print('---removing silence---')
        # no_silence = ProcessAudio.remove_silence(no_noise, min_silence_duration=min_sil, noise_threshold=n_t, sample_rate_hz=audio_obj.rate)

        print('---making chunks---')
        splits = ProcessAudio.split_audio(aud, chunk_length_s=c_l, sample_rate_hz=audio_obj.rate)
        split_recordings = splits

        # print('---removing unvoiced chunks---')
        # split_recordings = ProcessAudio.remove_unvoiced(splits, HNR_threshold=HNR, sr=audio_obj.rate)
        # console output

        print(
            f'SUMMARY [{count}/{input_len}] : Splitting file {file_path} into {len(split_recordings)} files (each {c_l}s)')
        count += 1

        # iterate through and save chunks to output dir
        for idx, recording in enumerate(split_recordings):
            file_stem = Path(file_path).stem
            new_file_path = file_stem + '_' + str(idx) + ".wav"
            dir_path = os.path.join(output_dir, new_file_path)
            # print(f'Saving to {dir_path}')
            scipy.io.wavfile.write(dir_path, audio_obj.rate, recording)

    output_len = len([name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name))])
    print(f'Completed preprocessing: {input_len} files converted to {output_len} files, saved in {output_dir}')


def process_commandline():
    parser = argparse.ArgumentParser(
        description='To preprocess audio data')

    # basic parameter arguments
    parser.add_argument('input_dir', default="processed_data/norm_CHILD_DATA",
                        help="Directory containing raw audio data")
    parser.add_argument('output_dir', default="processed_data/aud_chunks",
                        help="Path to save processed data")
    args = parser.parse_args()

    # Must have data directory and run title
    if not args.input_dir or not args.output_dir:
        parser.error('Must supply, input and output dir.')

    return args


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    try:
        os.makedirs(output_dir)
    except FileExistsError as e:
        print(e)
        answer = input("Do you want to continue? [y/n]:")
        if answer == 'n':
            sys.exit()

    process_and_save(input_dir, output_dir, c_l=0.2, HNR=50)


if __name__ == '__main__':
    main(process_commandline())
