# cleft-assessment-project
The code for my Msc thesis on automatic cleft-palate resonance assessment using CNNs. 

--- A work in progress ---

Current pipeline plan (and where it's located):
- Exactract audio from video (TODO)
- (Speaker diarization) (TODO)
- (Extract vowels/words) (partial - trimmer.py)
- Get meta data (meta_data.py)
- Processing audio (resample/channels/shift etc) (pre-process.py)
- (Remove noise) (TODO)
- Create spectrogram + labels (pre-process.py)
- (Mask) (pre-process.py)
- Create model and input data to CNN (CNN_pt.py)
- Print classification accuracy results! (CNN_pt.py)

Currently to be run as:

`python CNN_pt.py [dataset location] test [name of test run]`



### Credits
Adapted from [this tutorial]([url](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5)) by Ketan Doshi.
