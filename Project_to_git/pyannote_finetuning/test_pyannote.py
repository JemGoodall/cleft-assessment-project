from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio import Model
from pyannote.pipeline import Optimizer
from pyannote.database import FileFinder
from pyannote.database import get_protocol
from pyannote.audio import Pipeline
import sys
import os
from pathlib import Path

# Set up file finder and protocol for the UXTD data
preprocessors = {"audio": FileFinder()}
uxtd = get_protocol('MyDatabase.SpeakerDiarization.MyProtocol',
                    preprocessors=preprocessors)

# path to finetuned model
model_path = 'lightning_logs/version_3/checkpoints/epoch=9-step=3370.ckpt'

# create the pipeline with the pretrained and finetuned model
pretrained = Pipeline.from_pretrained("pyannote/speaker-diarization@2022.07")
finetuned_model = Model.from_pretrained(model_path, strict=False)
finetuned = SpeakerDiarization(segmentation=finetuned_model, embedding="speechbrain/spkrec-ecapa-voxceleb")

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
finetuned = finetuned.instantiate(init_params)

#  (pretrained.segmentation,  # pyannote/segmentation@2022.07
#       pretrained.segmentation_step,  # 0.1
#       pretrained.embedding,  # from speechbrain/spkrec-ecapa-voxceleb
#       pretrained.embedding_exclude_overlap,  # True
#       pretrained.clustering)  #HiddenMarkovModelClustering


def make_rttms_file(wav_file, pipeline, outfile_path):
    diarization = pipeline(wav_file, max_speakers=2)
    with open(outfile_path, "w") as rttm:
        diarization.write_rttm(rttm)


# iterate through wav_dir and create rttm files for single pipeline
def make_rttms_from_pipeline(wav_dir, pipeline, output_dir):
    for filename in os.listdir(wav_dir):
        wav_file = os.path.join(wav_dir, filename)
        file_stem = Path(wav_file).stem
        outfile_path = os.path.join(output_dir, file_stem+'.rttm')

        print(f'making rttm file from {wav_file}')
        make_rttms_file(wav_file, pipeline, outfile_path)

# create variables for directories
wav_dir = 'cleft_test_files/wavs'
pretrained_rttms_dir = 'cleft_test_files/pretrained_rttms'
finetuned_rttms_dir = 'cleft_test_files/finetuned_rttms'
ft_optimised_dir = 'cleft_test_files/ft_optimised_rttms'

print('--- pretrained! ---')
make_rttms_from_pipeline(wav_dir, pipeline=pretrained, output_dir=pretrained_rttms_dir)
#
# print('--- finetuned! ---')
# make_rttms_from_pipeline(wav_dir, pipeline=finetuned, output_dir=finetuned_rttms_dir)

# print('--- finetuned_optimised! ---')
# make_rttms_from_pipeline(wav_dir, pipeline=finetuned_optimised, output_dir=ft_optimised_dir)

