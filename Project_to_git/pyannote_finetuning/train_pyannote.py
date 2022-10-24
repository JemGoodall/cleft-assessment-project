from pyannote.database import FileFinder
from pyannote.database import get_protocol
from pyannote.audio.tasks import Segmentation
from pyannote.core import notebook
from pyannote.audio import Model
from pyannote.audio import Inference
from copy import deepcopy
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger

import sys

# ------- GET DER --------
def _test(model, protocol, subset="test"):
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


def get_der(model, protocol, subset, name="model"):
    der = _test(model=model, protocol=protocol, subset=subset)
    print(f"Local DER {name} = {der * 100:.1f}%")


# ----- FINE TUNING ------
def train(protocol, pretrained):
    seg_task = Segmentation(protocol, duration=1.0, max_num_speakers=2)
    finetuned = deepcopy(pretrained)
    finetuned.task = seg_task
    trainer = pl.Trainer(gpus=1, max_epochs=10)
    trainer.fit(finetuned)

    return finetuned


def main():
    preprocessors = {"audio": FileFinder()}
    uxtd = get_protocol('MyDatabase.SpeakerDiarization.MyProtocol',
                        preprocessors=preprocessors)
    # load a pretrained model
    pretrained = Model.from_pretrained("pyannote/segmentation")

    # make CSV logger and train
    # logger = CSVLogger(save_dir="lightning_logs", name="trials")
    # logger = TensorBoardLogger(save_dir="lightning_logs", name="trials")
    finetuned = train(uxtd, pretrained)

    print("--- getting DER of finetuned model ---")
    get_der(finetuned, uxtd, subset="test", name="finetuned")  # eval pretrained


if __name__ == '__main__':
    main()

# -------------- EVALUATION CODE -----------------
# https://github.com/pyannote/pyannote-metrics/blob/master/notebooks/pyannote.metrics.diarization.ipynb
# https://github.com/pyannote/pyannote-core/issues/17
