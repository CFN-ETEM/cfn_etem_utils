import argparse
import glob
import json, math
import os, socket
import re
import cv2
from itertools import chain

from ict.particle_tracking.io import load_dm4_file

import ipyparallel as ipp


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

def get_parallel_map_func():
    c = ipp.Client(connection_info="ipypar/security/ipcontroller-client.json")

    print(c.ids)

    with c[:].sync_imports():
        import cv2, os, socket
        from ict.particle_tracking.io import load_dm4_file


    return c[:].map_sync, len(c.ids)


def init_params():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="The folder constains DM4 files",
                        required=True,
                        type=lambda x: is_valid_directory(parser, x))
    parser.add_argument("-t", "--to", help="The folder to hold the PNG files",
                        required=True,
                        type=lambda x: is_valid_directory(parser, x))
    parser.add_argument("-l", "--list_save", help="Save the file list",
                        action='store_true')
    parser.add_argument("-p", "--parallel", help="Run the job in parallel mode",
                        action='store_true')
    args = parser.parse_args()
    source_dir = os.path.abspath(args.source)
    dest_dir = os.path.abspath(args.to)
    list_save = args.list_save
    parallel_run = args.parallel
    in_file_list = list(chain(*[glob.glob("{}/Capture{}/**/*.dm4".format(source_dir, "[0-9]" * n), recursive=True)
                                for n in range(1, 4)]))
    if len(in_file_list) == 0:
        in_file_list = glob.glob("{}/**/*.dm4".format(source_dir), recursive=True)
    if len(in_file_list) == 0:
        print("Error: no files found")
        exit(1)
    in_file_list = sorted(in_file_list, key=lambda x: (
        int(re.search(r'Capture(\d+).*', x).group(1)),
        x))

    return parallel_run, source_dir, dest_dir, in_file_list, list_save


def sub_job(source_dir, dest_dir, in_file_list, list_save):
    if list_save:
        with open("list_of_source_files/list_of_source_files.pid{}.{}.txt".
                          format(os.getpid(), socket.gethostname()), "w") as f:
            f.writelines([fn + '\n' for fn in in_file_list])
    small_files = []
    for in_fn in in_file_list:
        img = load_dm4_file(in_fn)
        out_fn = os.path.abspath(in_fn).replace(source_dir, dest_dir).replace("dm4", "png")
        if not os.path.exists(os.path.dirname(out_fn)):
            os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        cv2.imwrite(out_fn, img)
        if os.path.getsize(out_fn) / (1024*1024) < 1.0:
            small_files.append(out_fn)

    if len(small_files) > 0:
        with open("small_files.pid{}.{}.txt".format(os.getpid(), socket.gethostname()), "w") as f:
            f.writelines([fn + '\n' for fn in small_files])


if __name__ == '__main__':
    parallel_run, source_dir, dest_dir, full_in_file_list, list_save = init_params()

    if parallel_run:
        map, nprocesses = get_parallel_map_func()
    else:
        map, nprocesses = map, 1

    print("running with {} processes".format(nprocesses))

    if list_save:
        os.makedirs("list_of_source_files", exist_ok=True)
    nfiles_per_process = math.ceil(len(full_in_file_list) / nprocesses)
    ifl_chunks = [full_in_file_list[i * nfiles_per_process: (i + 1) * nfiles_per_process]
                  for i in range(nprocesses)]
    map(sub_job,
        [source_dir] * nprocesses,
        [dest_dir] * nprocesses,
        ifl_chunks,
        [list_save] * nprocesses)
