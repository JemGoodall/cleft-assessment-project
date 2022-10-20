import argparse


def process_commandline():
    parser = argparse.ArgumentParser(
        description='CNN -- work in progress --')

    # basic parameter arguments
    parser.add_argument('data_directory', default="../spec_data/testing_script/train_data",
                        help="Directory containing data (tensors)")
    parser.add_argument('run_title', nargs='?', default='training X',
                        help="Set the run title (saves to graph and logs)")
    parser.add_argument('--test', '-t', action="store_true", default=False,
                        help="Whether this is a test (changes meta_data action)")
    parser.add_argument('--meta_data', '-md', default="../spec_data/testing_script/train_data",
                        help="meta_data file if different from data_directory")

    # Optional arguments for model/data:
    parser.add_argument('--epochs', '-e', default=100, type=int,
                        help="Set the number of epochs (default 100)")
    parser.add_argument('--learningrate', '-lr', default=0.1, type=float,
                        help="Float specifying the learning rate")
    parser.add_argument('--batchsize', '-b', action="store", default=64, type=int,
                        help="Define the batch size")
    parser.add_argument('--graphoff', '-g', action="store_true", default=False,
                        help="Do not make a graph")
    parser.add_argument('--norm_type', '-nt', default="global", choices=['mels', 'global'],
                        help="Decide how to norm the data (either globally and by mels")
    parser.add_argument('--weights', '-w', action="store_false", default=True,
                        help="Use raw class weights rather than scaled back weights")
    parser.add_argument('--mix_patients', '-mix', action="store_true", default=False,
                        help="Don't separate speakers between train and test dataset")
    parser.add_argument('--patient_ids', action="store",
                        help="temp list")

    # Optional arguments for lrcn:
    parser.add_argument('--lrcn', action="store_true", default=False,
                        help="Use LRCN model")
    parser.add_argument('--input_dims', '-id', action="store", default=64, type=int,
                        help="Define input dims for LSTM (CNN output)")
    parser.add_argument('--hidden_dims', '-hd', action="store", default=128, type=int,
                        help="Define the hidden dimensions of LSTM")
    parser.add_argument('--dropout_rate', '-dr', action="store", default=0.4, type=float,
                        help="Define the dropout rate in LSTM")
    parser.add_argument('--maxlr', '-mlr', action="store", default=0.0001, type=float,
                        help="Define the max learning rate")




    # # Optional arguments for data:
    # parser.add_argument('--SpecType', '-sp', action="store", default='Mel', choices=['Mel', 'CQT'],
    #                     help="The type of spectrogram used")
    # parser.add_argument('--mask', '-ma', action="store_true", default=False,
    #                     help="Add masking to the data")
    # parser.add_argument('--shift', '-sh', action="store_true", default=False,
    #                     help="Enable shifting audio")
    # parser.add_argument('--mels', '-me', default=64, type=int,
    #                     help="Define the number of freq bins")
    # parser.add_argument('--fft', '-f', default=256, type=int,
    #                     help="Define the length of the fft window in samples")
    # parser.add_argument('--samplerate', '-sa', default=8000, type=int,
    #                     help="Define the sample rate")

    args = parser.parse_args()

    # Must have data directory and run title

    if not args.data_directory or not args.run_title:
        parser.error('Must supply both data directory and run title')

    return args
