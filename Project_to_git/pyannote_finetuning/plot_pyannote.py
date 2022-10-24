from pyannote.database import FileFinder
from pyannote.database import get_protocol
from pyannote.core import Segment
from pyannote.audio.tasks import Segmentation
from pyannote.core import notebook
from pyannote.core.notebook import repr_feature, repr_segment, repr_timeline, repr_annotation
from pyannote.audio import Model
from pyannote.audio import Inference
from pyannote.audio.pipelines import SpeakerDiarization

# initialize matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib.pyplot import figure

import sys


# Set up visualisation - crop to 1 min, starting 240s
# one_minute = Segment(240, 300)
# notebook.crop = one_minute


def _save_pyannote_fig(repr_funct, window, name):
    # save pyannote instances - repr_funct should be one of the following:
    # repr_feature, repr_segment, repr_timeline, repr_annotation
    png_data = repr_funct(window)
    with open(f"plots/{name}.png", "wb") as f:
        f.write(png_data)


def save_fig(image, name):
    figure, ax = plt.subplots()
    mpl.use('Agg')
    notebook.width = 10
    plt.rcParams['figure.figsize'] = (notebook.width, 2)
    notebook.plot_annotation(image, ax=ax, time=True, legend=True)
    # save to file
    figure.savefig(f'plots/{name}.png')


def save_annotation(annotation, name):
    notebook.crop = Segment(0, 32)
    # plot
    plt.rcParams["figure.figsize"] = (20, 2)
    figure, ax = plt.subplots(constrained_layout=True)
    figure.suptitle(name)
    notebook.plot_annotation(annotation, ax=ax, time=True, legend=True)
    ax.set_ylabel('Speaker segments')
    # save to file
    figure.savefig(f'plots/{name}.png')


# ---- MAKE AND SAVE PLOTS -----
# compare the pretrained model with the target labels
def get_graph_inference(test_file, model, file_name):
    # visualise how it performs on a test file
    # step = the sliding window step, a ratio of its duration ie. 0.1 = 90% overlap -
    predicted_spk = Inference(model, duration=1, step=0.2)(test_file)  # returns slidingWindowFeature
    _save_pyannote_fig(repr_feature, predicted_spk, str(file_name))


def get_graph_target(test_file, name):
    # the target output looks like this:
    target_spk = test_file["annotation"].discretize(notebook.crop, resolution=0.010) # returns slidingWindowFeature
    _save_pyannote_fig(repr_feature, target_spk, 'target_spk')


def load_model(model_path):
    finetuned_model = Model.from_pretrained(model_path, strict=False)
    return finetuned_model


# ------- GET DER --------
def _test_local(model, protocol, subset="test"):
    from pyannote.audio.utils.signal import binarize
    from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
    from pyannote.audio.pipelines.utils import get_devices

    (device,) = get_devices(needs=1)
    metric = DiscreteDiarizationErrorRate()
    files = list(getattr(protocol, subset)())

    inference = Inference(model, device=device, duration=1.0, step=0.2)

    for file in files:
        reference = file["annotation"]
        hypothesis = binarize(inference(file))
        uem = file["annotated"]
        _ = metric(reference, hypothesis, uem=uem)

    return abs(metric)


def get_local_der(model, protocol, subset, name="model"):
    der = _test_local(model=model, protocol=protocol, subset=subset)
    print(f"Local DER {name} = {der * 100:.1f}%")


def get_total_der(pipeline, protocol, name="model"):
        from pyannote.metrics.detection import DetectionErrorRate
        metric = DetectionErrorRate()

        for file in protocol.test():
            # apply the voice activity detection pipeline
            print(file["uri"])
            speech = pipeline(file)

            # evaluate its output
            _ = metric(
                file['annotation'],  # this is the reference annotation
                speech,  # this is the hypothesized annotation
                uem=file['annotated'])  # this is the part of the file that should be evaluated

        # aggregate the performance over the whole test set
        detection_error_rate = abs(metric)
        print(f'Pipeline DER for {name} = {detection_error_rate * 100:.1f}%')


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


def main():
    NAME = 'test'
###############################
# Code for plotting annotation (horizonatal segments)
# Requires alternative YAML file, kinda hacky, but plot is nicer

#     preprocessors = {"cleft_test_files": FileFinder()}
#     cleft = get_protocol('MyDatabase.SpeakerDiarization.MyProtocol',
#                         preprocessors=preprocessors)

#     list_of_text_files = list(cleft.test())
#     test_file = list_of_text_files[0]
#     print(f'test_file: {test_file["uri"]}')

#     print('Creating segmentation plots')
#     save_annotation(test_file['annotation'], NAME)

#     sys.exit()

###############################

    preprocessors = {"audio": FileFinder()}
    uxtd = get_protocol('MyDatabase.SpeakerDiarization.MyProtocol',
                        preprocessors=preprocessors)

    # MODEL TO TEST:
    model_path = 'lightning_logs/version_0/checkpoints/epoch=4-step=1685.ckpt'

    print(f'loading model: {model_path}')
    finetuned = load_model(model_path)

    print(f'getting Local DER...')   # get DER of model on test set
    get_local_der(finetuned, uxtd, subset="test", name=NAME)

    # make pipeline
    print('Making pipeline..')
    finetuned_pipeline = SpeakerDiarization(segmentation=finetuned,embedding="speechbrain/spkrec-ecapa-voxceleb")
    # give default params to finetuned model
    print('Instantiating params...')
    finetuned_pipeline = finetuned_pipeline.instantiate(init_params)

    # print('getting overall DER...')
    get_total_der(finetuned_pipeline, uxtd, name=NAME)

    test_file = next(uxtd.test())  # load a test file
    print(f'test_file: {test_file["uri"]}'
    print('Creating segmentation plots of target and prediction...')
    get_graph_inference(test_file, model=finetuned, file_name=NAME)
          
    print('loading pretrained model..')
    pretrained = Model.from_pretrained("pyannote/segmentation")

    get_graph_inference(test_file, model=pretrained, file_name='pretrained_102')
    get_graph_target(test_file)
    save_annotation(test_file["annotation"], 'annotation_target')

if __name__ == '__main__':
    main()
