# cleft-assessment-project
The code for my Msc thesis on automatic cleft-palate hypernasality assessment using CNN and LRCN models. 
LRCNs (Long short-term recurrent convolutional networks) combine CNN feature extraction with LSTM sequential processing, making them appropriate for use in speech classification tasks with spectrogram input. 

--- A more-or-less finished project (needs tidying) ---

Data Preprocessing:
- Extract audio from video and normalise dB (extract_audio.sh, normalise_db.sh)
- Speaker diarization (diarization.py)
- Split into x ms segments and remove semgents with low HNR (preprocess.py)
- Create spectrograms (make_spec_data.py)

Train and test (full_traintest.py):
- Create model (either CNN or LRNC, in model.py)
- Split data depending on patient input, or random. 
- Create dataloaders depending on model (in make_dataset.py)
- Train and save results
- Test and save results

Bonus! Finetuning pyannote:
- (NEED TO ADD FILE TO GIT)
