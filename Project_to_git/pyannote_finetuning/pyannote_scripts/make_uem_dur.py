"""
For making .uem files out of .lab files (from ULTRAX)
use:
python make_uem.py [input_dir] [output_dir]

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

    make_uem(input_dir, out_dir, total_files)


def make_uem_files(out_dir, data_set):
    out_path = os.path.join(out_dir, f'{data_set}.uem')
    return out_path


def make_uem(input_dir, out_dir, total_files):
    # create uem files and get path
    train_path = make_uem_files(out_dir, 'train')
    dev_path = make_uem_files(out_dir, 'development')
    test_path = make_uem_files(out_dir, 'test')

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
        print(f'[{idx}/{total_files}] Writing uem for {file_path}, sending to {outfile_path}')
        idx += 1

        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
            if lines:  # some files are empty...?
                first = lines[0]
                last = lines[-1]
                tbeg = int(first.split()[0]) / (10 ** 7)  # get the start time of first line
                tend = int(last.split()[1]) / (10 ** 7)  # get the end dur of last line
                # need to divide to get value in seconds
                if tend - tbeg > 2: # only use audio that is longer than 2 seconds
                    new_line = f'{file_stem} NA {tbeg:.3f} {tend:.3f}\n'

                    with open(outfile_path, 'a') as out_file:
                        out_file.write(new_line)
                else:
                    print(f'file {file_stem} too short, skipping.')


if __name__ == '__main__':
    main()
