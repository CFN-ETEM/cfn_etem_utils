import argparse
import glob
import json
import os
import re
from itertools import chain

from cfntem.particle_tracking.io import average_file_list


def is_valid_directory(parser, arg):
    if not os.path.isdir(arg):
        parser.error('The directory {} does not exist!'.format(arg))
    else:
        return arg

def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        return arg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="The folder constains original image",
                        required=True,
                        type=lambda x: is_valid_directory(parser, x))
    parser.add_argument("-t", "--to", help="The folder to hold the averaged image",
                        required=True,
                        type=lambda x: is_valid_directory(parser, x))
    parser.add_argument("-w", "--window_size", help="The window size to take average",
                        default=9,
                        type=int)
    parser.add_argument("-l", "--list_save", help="Save the file list",
                        action='store_true')

    args = parser.parse_args()

    source_dir = os.path.abspath(args.source)
    dest_dir = os.path.abspath(args.to)
    list_save = args.list_save
    window_size = args.window_size
    half_window_size = window_size // 2

    in_file_list = list(chain(*[glob.glob("{}/Capture{}/**/*.png".format(source_dir, "[0-9]"*n), recursive=True)
            for n in range(1, 4)]))
    if len(in_file_list) == 0:
        in_file_list = glob.glob("{}/**/*.png".format(source_dir), recursive=True)
    if len(in_file_list) == 0:
        print("Error: no files found")
        exit(1)
    in_file_list = sorted(in_file_list, key=lambda x:(
        int(re.search(r'Capture(\d+).*', x).group(1)),
        x))

    if list_save:
        with open("list_of_source_files.txt", "w") as f:
            f.writelines([fn + '\n' for fn in in_file_list])


    error_files = []
    total_num_files = len(in_file_list)
    for focus_index, center_fn in enumerate(in_file_list):
        if (focus_index - half_window_size) % window_size == 0 and half_window_size <= focus_index < total_num_files - half_window_size:
            out_fn = os.path.abspath(center_fn).replace(source_dir, dest_dir).replace("dm4", "png")
            _, err_info = average_file_list(in_file_list, focus_index, window_size, out_fn)
            if err_info is not None:
                error_files.append(err_info)

    if len(error_files) > 0:
        with open("error_files.json", "w") as f:
            json.dump(error_files, f, indent=4, sort_keys=True)
    else:
        print("All the files are processed successfully")


if __name__ == '__main__':
    main()
