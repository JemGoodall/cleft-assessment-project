"""
For making .lst files out of .uem files
use:
python make_lst_dur.py [OUTDIR]


"""

import sys
import os
from pathlib import Path


def main():
    try:
        out_dir = sys.argv[1]
    except IndexError:
        print("Please provide output_dir")
        sys.exit()

    uem_dir = '../uems'

    make_lst(uem_dir, out_dir)


def make_lst(uem_dir, out_dir):
    for filename in sorted(os.listdir(uem_dir)):
        outfile_path = os.path.join(out_dir, f'{Path(filename).stem}.lst')
        file_path = os.path.join(uem_dir, filename)
        with open(file_path, 'r') as uem_file:
            with open(outfile_path, 'a') as lst_file:
                for line in uem_file:

                    uri = line.split()[0]
                    lst_file.write(uri + '\n')
                    print(f'writing {uri} to {outfile_path}')
        # file_stem = Path(file_path).stem  # get the file name without the extension
        #  = int(file_stem[:2])  # get the patient num from file name
        #
        #
        # # write console output
        # print(f'[{idx}] Writing lst for {file_stem}, sending to {outfile_path}')
        # idx += 1
        #
        # with open(outfile_path, 'a') as out_file:
        #     out_file.write(file_stem + '\n')  # CHECK might need the file extension
        #


if __name__ == '__main__':
    main()
