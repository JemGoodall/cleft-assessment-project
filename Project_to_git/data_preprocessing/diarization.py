'''
For taking a dir of audio files, splitting the audio based on speaker,
and saving new audio files to a different dir
e.g. if an .wav file contains 3 speakers, it will be split into 3 different .wav files renamed with '_idx'

OR splitting the audio based on speaker and utterance.
e.g. filename_uttNum_speakerNum.wave

use:
python diarisation.py [input_dir] [output_dir] ['lrcn' or 'cnn']

NB all files will be put into output_dir --> use mv_dia_wavs.sh after to split into two dirs INSIDE of output_dir
NB MAKE SURE TO CHECK diarization = pipeline(file_path, max_speakers=1) LINE!!!!!!!
'''
from pydub import AudioSegment
import os
from pathlib import Path
import sys
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio import Model


def main():
    try:
        input_dir = sys.argv[1]
    except IndexError:
        print("Please provide a input_dir of wav files")
        sys.exit()

    try:
        output_dir = sys.argv[2]
    except IndexError:
        print("Please provide output_dir")
        sys.exit()

    try:
        split_type = sys.argv[3]
    except IndexError:
        print("Please specify if 'lrcn' or 'cnn'")
        sys.exit()

    split_audio_dir(input_dir, output_dir, load_finetuned_model(), split_type)


def load_finetuned_model():
    # path to finetuned model
    model_path = '/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1509050_Jemima_Goodall/ULTRAX/lightning_logs/version_3/checkpoints/epoch=9-step=3370.ckpt'

    # load model and add it to the speaker diarization pipeline
    print(f'Loading model from {model_path}...')
    finetuned_model = Model.from_pretrained(model_path, strict=False)
    finetuned_pipeline = SpeakerDiarization(segmentation=finetuned_model, embedding="speechbrain/spkrec-ecapa-voxceleb")

    init_params = {
        "segmentation_onset": 0.58,
        "clustering": {
            "single_cluster_detection": {
                "quantile": 0.05,
                "threshold": 1.15,
            },
            "covariance_type": "diag",
            "threshold": 0.35,
        },
    }

    # give default params to finetuned model
    finetuned_pipeline = finetuned_pipeline.instantiate(init_params)

    return finetuned_pipeline


# function to take diarization and return separate AudioSegments
def split_audio(diarization, audio):
    speaker_00 = AudioSegment.empty()
    speaker_01 = AudioSegment.empty()
    # speaker_02 = AudioSegment.empty()
    splits = [speaker_00, speaker_01]
    for turn, speaker_id, _ in diarization.itertracks(yield_label=True):
        splits[speaker_id] += audio[turn.start * 1000:turn.end * 1000]
        # print(f"start={turn.start:.1f}s, stop={turn.end:.1f}s speaker_{speaker_id} == {_}")

    return splits


# takes diarization and returns AudioSegment for each utterance (NOT 1 audio per speaker!)
def split_audio_per_utterance(diarization, audio):
    dict_splits = {0: [], 1: []}
    for turn, speaker_id, _ in diarization.itertracks(yield_label=True):
        dict_splits[speaker_id].append(audio[turn.start * 1000:turn.end * 1000])

    return dict_splits


# takes a list of audio split by speaker
def save_audio(splits, outfile_path):
    for i, audio in enumerate(splits):
        if len(audio) != 0.0:
            # print(f'Saving {outfile_path}_{i} for speaker_{i} is {len(audio) / 1000}s long')
            audio.export(f'{outfile_path}_{i}.wav', format='wav')


# takes a dict of audio with separate utterances per speaker
def save_audio_per_utterance(dict_splits, outfile_path):
    for s, speaker in enumerate(dict_splits):
        for u, utterance in enumerate(dict_splits[speaker]):
            if len(utterance) > 200:  # don't include utts shorter than 200ms
                utterance.export(f'{outfile_path}_{u}_{s}.wav', format='wav')


def split_audio_dir(input_dir, output_dir, pipeline, split_type='cnn'):
    total_files = len([name for name in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, name))])
    idx = 1
    for filename in sorted(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, filename)
        file_stem = Path(file_path).stem
        outfile_path = os.path.join(output_dir, file_stem)

        # load the audio
        audio = AudioSegment.from_wav(file_path)

        # apply pretrained pipeline
        print(f'Analysing speakers...')
        # diarization = pipeline(file_path, min_speakers=1, max_speakers=2)
        diarization = pipeline(file_path, max_speakers=2)  # for just removing silence...

        # split and save the audio based on speaker
        print(f'[{idx}/{total_files}] Splitting {filename}')
        if split_type == 'cnn':
            splits = split_audio(diarization, audio)
            save_audio(splits, outfile_path)
            # print(f'Original audio length: {len(AudioSegment.from_wav(file_path))/ 1000}s')
        elif split_type == 'lrcn':
            dict_splits = split_audio_per_utterance(diarization, audio)
            save_audio_per_utterance(dict_splits, outfile_path)
        idx+=1


if __name__ == '__main__':
    main()
