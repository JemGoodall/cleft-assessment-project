"""
For making .lst files out of .lab files (from ULTRAX)
use:
python make_lst.py [input_dir] [output_dir]

input dir copy paste: ../uxtd/speaker_labels/lab


"""

import sys
import os
from pathlib import Path


def main():
    try:
        input_dir = sys.argv[1]
    except IndexError:
        print("Please provide a input_dir of .lab files")
        sys.exit()

    try:
        out_dir = sys.argv[2]
    except IndexError:
        print("Please provide output_dir")
        sys.exit()

    total_files = len([name for name in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, name))])

    make_lst(input_dir, out_dir, total_files)


def get_path(out_dir, data_set):
    out_path = os.path.join(out_dir, f'{data_set}.lst')
    return out_path


def make_lst(input_dir, out_dir, total_files):
    # write first line of the rttms file
    train_path = get_path(out_dir, 'train')
    dev_path = get_path(out_dir, 'development')
    test_path = get_path(out_dir, 'test')

    idx = 1
    for filename in sorted(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, filename)
        file_stem = Path(file_path).stem  # get the file name without the extension
        patient_num = int(file_stem[:2])  # get the patient num from file name

        # set the outfile to either train, dev, or test
        if patient_num < 36:
            outfile_path = train_path
        elif 35 < patient_num < 47:
            outfile_path = dev_path
        else:
            outfile_path = test_path

        # write console output
        print(f'[{idx}/{total_files}] Writing lst for {file_stem}, sending to {outfile_path}')
        idx += 1

        with open(outfile_path, 'a') as out_file:
            out_file.write(file_stem + '\n')  # CHECK might need the file extension


if __name__ == '__main__':
    main()
