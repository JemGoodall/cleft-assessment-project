import os
import shutil

rootdir = 'core-uxtd/core'
newdir = 'audio'
if not os.path.exists(newdir):
    os.mkdir(newdir)
txt_path = 'wav_path_list5.txt'
with open(txt_path, 'a') as txt:
    for subdir, dirs, files in sorted(os.walk(rootdir)):
        samples = subdir[15:18]  # get the num and gender
        dir_path = os.path.join(newdir, samples)

        print(f'writing {dir_path}')
        txt.write(f'- {dir_path}/{{uri}}.wav\n')

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        for file in files:
            if '.wav' in file:
                old_file_path = os.path.join(subdir, file)
                new_file_name = samples[:2] + samples[2] + '-' + file
                new_file_path = os.path.join(dir_path, new_file_name)
                # shutil.copy(old_file_path, new_file_path)
                # print(f'moving {old_file_path} to {new_file_path}')

                # write to txt file for later use in yaml
                # print(f'writing {new_file_path}')
                # txt.write(f'- ./{new_file_path}\n')

