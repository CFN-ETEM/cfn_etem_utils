import glob, os, math
import argparse
import json
import re
from itertools import chain
from cfntem.particle_tracking.io import binarize_file_list

from cfntem.particle_tracking.image_processing import ImageEnhance, ImageBinarization, ImageMerge

import ipyparallel as ipp
import socket


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
        import glob, os
        import argparse
        import json
        import re
        import socket
        from itertools import chain
        from cfntem.particle_tracking.io import binarize_file_list


    return c[:].map_sync, len(c.ids)

def sub_job(in_file_list, source_dir, dest_dir, circled_dir, interval, half_window_size, merge_processor, list_save,
            internal_error_estimation, circle_frequency, raw_img_dir):
    if list_save:
        with open("list_of_source_files/list_of_source_files.pid{}.{}.txt".format(os.getpid(), socket.gethostname()), "w") as f:
            f.writelines([fn + '\n' for fn in in_file_list])
    error_list = dict()
    total_num_files = len(in_file_list)
    incompatible_files = []
    for focus_index, center_fn in enumerate(in_file_list):
        if (focus_index - interval // 2) % interval == 0 and \
                half_window_size <= focus_index < total_num_files - half_window_size:
            out_fn = os.path.abspath(center_fn).replace(source_dir, dest_dir).replace("dm4", "png")
            if circled_dir is not None and (focus_index - interval // 2) % circle_frequency == 0:
                circled_fn = os.path.abspath(center_fn).replace(source_dir, circled_dir).replace("dm4", "png")
                raw_fn = os.path.abspath(center_fn).replace(source_dir, raw_img_dir)
            else:
                circled_fn = None
                raw_fn = None
            try:
                _, _, trust_ratio = binarize_file_list(in_file_list, focus_index, [merge_processor],
                                                       out_fn, circled_fn, internal_error_estimation,
                                                       merge_only=True, raw_file=raw_fn)
                error_list[center_fn] = trust_ratio
            except:
                incompatible_files.append(center_fn)

    if len(incompatible_files) > 0:
        with open("incompatible_files.pid{}.{}.txt".format(os.getpid(), socket.gethostname()),
                  "w") as f:
            f.writelines([fn + '\n' for fn in incompatible_files])

    return error_list


def init_params():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="The folder constains binarized image",
                        required=True,
                        type=lambda x: is_valid_directory(parser, x))
    parser.add_argument("-t", "--to", help="The folder to hold the binarized image",
                        required=True,
                        type=lambda x: is_valid_directory(parser, x))
    parser.add_argument("-c", "--circled", help="The folder to hold image with marked original file",
                        required=False,
                        type=lambda x: is_valid_directory(parser, x))
    parser.add_argument("--raw_dir", help="The folder to hold original file",
                        required=False,
                        type=lambda x: is_valid_directory(parser, x))
    parser.add_argument("--circle-frequency", help="The frequency to write a circled file, in number of windows",
                        default=10,
                        type=int)
    parser.add_argument("-m", "--merge-param-file", help="JSON file for frame merge parameter",
                        required=False,
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-l", "--list_save", help="Save the file list",
                        action='store_true')
    parser.add_argument("-p", "--parallel", help="Run the job in parallel mode",
                        action='store_true')
    parser.add_argument("--internal-error-estimation", help="Estimate the internal detection error",
                        action='store_true')
    args = parser.parse_args()
    source_dir = os.path.abspath(args.source)
    dest_dir = os.path.abspath(args.to)
    circled_dir = os.path.abspath(args.circled) if args.circled else None
    list_save = args.list_save
    parallel_run = args.parallel
    internal_error_estimation = args.internal_error_estimation
    raw_dir = args.raw_dir
    merge_processor = ImageMerge(args.merge_param_file)
    window_size = merge_processor.params["window_size"]
    interval = window_size
    half_window_size = window_size // 2
    assert interval >= window_size
    circle_frequency = args.circle_frequency * window_size
    in_file_list = list(
        chain(*[glob.glob("{}/Capture{}/**/*.{}".format(source_dir, "[0-9]" * n, pic_format), recursive=True)
                for n in range(1, 4) for pic_format in ["dm4", "png"]]))
    if len(in_file_list) == 0:
        in_file_list = list(chain(*[glob.glob("{}/**/*.{}".format(source_dir, pic_format), recursive=True)
                                    for pic_format in ["dm4", "png"]]))
    if len(in_file_list) == 0:
        print("Error: no files found")
        exit(1)
    in_file_list = sorted(in_file_list, key=lambda x: (
        int(re.search(r'Capture(\d+).*', x).group(1)),
        x))
    return parallel_run, source_dir, dest_dir, circled_dir, list_save, interval, internal_error_estimation, \
           merge_processor, half_window_size, in_file_list, circle_frequency, raw_dir


def main():
    parallel_run, source_dir, dest_dir, circled_dir, list_save, interval, internal_error_estimation, merge_processor, \
            half_window_size, full_in_file_list, circle_frequency, raw_dir = init_params()
    if parallel_run:
        map, nprocesses = get_parallel_map_func()
    else:
        map, nprocesses = map, 1

    nfiles_per_process = math.ceil(len(full_in_file_list) / (nprocesses * interval)) * interval
    ifl_chunks = [full_in_file_list[i * nfiles_per_process: (i + 1) * nfiles_per_process]
                  for i in range(nprocesses)]
    if not os.path.exists("list_of_source_files") and list_save:
        os.makedirs("list_of_source_files")
    el_list = map(sub_job, ifl_chunks,
                  [source_dir] * nprocesses,
                  [dest_dir] * nprocesses,
                  [circled_dir] * nprocesses,
                  [interval] * nprocesses,
                  [half_window_size] * nprocesses,
                  [merge_processor] * nprocesses,
                  [list_save] * nprocesses,
                  [internal_error_estimation] * nprocesses,
                  [circle_frequency] * nprocesses,
                  [raw_dir] * nprocesses)

    if internal_error_estimation:
        iee = dict()
        for d in el_list:
            iee.update(d)
        with open("internal_errs.json", "w") as f:
            json.dump(iee, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
