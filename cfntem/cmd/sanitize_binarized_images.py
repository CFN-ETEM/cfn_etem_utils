#! /usr/bin/env ipython
import os, json, re, shutil, math
import numpy
import argparse
import glob
from itertools import chain
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
        import re, os, shutil
        import numpy

    return c[:].map_sync, len(c.ids)


def init_params():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="The folder constains binary image",
                        required=True,
                        type=lambda x: is_valid_directory(parser, x))
    parser.add_argument("-t", "--to", help="The folder to hold sanitized image",
                        required=True,
                        type=lambda x: is_valid_directory(parser, x))
    parser.add_argument("-e", "--internal_error_file", help="The file for internal error estimation information",
                        required=True,
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-n", "--neighbours", help="Number of neighbouring frame to delete",
                        default=0,
                        type=int)
    parser.add_argument("-c", "--criteria", help="The file for sanitize criteria",
                        required=True,
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-p", "--parallel", help="Run the job in parallel mode",
                        action='store_true')
    args = parser.parse_args()
    source_dir = os.path.abspath(args.source)
    sani_dir = os.path.abspath(args.to)
    internal_error_file = args.internal_error_file
    cri_file = args.criteria
    neighbours = args.neighbours
    parallel_run = args.parallel
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
    good_dir = os.path.abspath(os.path.join(sani_dir, "good_pic"))
    poor_dir = os.path.abspath(os.path.join(sani_dir, "poor_pic"))
    with open(cri_file, "r") as f:
        err_criteria = json.load(f)
    with open(internal_error_file, "r") as f:
        err_list = json.load(f)

    return parallel_run, source_dir, neighbours, in_file_list, good_dir, poor_dir, err_criteria, err_list


def sub_job(source_dir, neighbours, in_file_list, good_dir, poor_dir, err_criteria, err_list):
    good_err_stats = dict()
    poor_err_stats = dict()
    for avg_fn, err_data in err_list.items():
        crit = []
        err_value = []
        for k, v in err_criteria.items():
            crit.extend(v)
            err_value.extend(err_data[k])
        crit = numpy.array(crit)
        err_value = numpy.abs(numpy.array(err_value))
        if (err_value <= crit).all():
            dest_dir = good_dir
            good_err_stats[avg_fn] = err_data
        else:
            dest_dir = poor_dir
            poor_err_stats[avg_fn] = err_data
        for sub_dir in ["bin_pic", "marked_pic"]:
            center_i = in_file_list.index(os.path.abspath(re.sub(".*pic", source_dir, avg_fn)))
            for i in range(-neighbours, neighbours + 1):
                if center_i + i >= len(in_file_list) and sub_dir == "marked_pic":
                    break
                in_fn = in_file_list[center_i + i]
                dest_fn = os.path.abspath(re.sub(".*pic", dest_dir + "/" + sub_dir, in_fn))
                src_fn = os.path.abspath(re.sub(".*pic", source_dir.replace("bin_pic", sub_dir), in_fn))
                if not os.path.exists(src_fn):
                    if i == 0 and sub_dir == "bin_pic":
                        raise ValueError("Even center file {} doesn't exist".format(src_fn))
                    continue
                sd = os.path.dirname(dest_fn)
                if not os.path.exists(sd):
                    os.makedirs(sd)
                shutil.copy(src_fn, dest_fn)
    return good_err_stats, poor_err_stats


def main():
    parallel_run, source_dir, neighbours, in_file_list, good_dir, poor_dir, err_criteria, err_list \
        = init_params()

    if parallel_run:
        map, nprocesses = get_parallel_map_func()
    else:
        map, nprocesses = map, 1

    interval = 1 + neighbours * 2
    el_tuples = sorted(err_list.items())
    nfiles_per_process = math.ceil(len(el_tuples) / nprocesses)
    el_chunks = [dict(el_tuples[i * nfiles_per_process: (i + 1) * nfiles_per_process])
                 for i in range(nprocesses)]

    gp_err_list = map(sub_job,
                      [source_dir] * nprocesses,
                      [neighbours] * nprocesses,
                      [in_file_list] * nprocesses,
                      [good_dir] * nprocesses,
                      [poor_dir] * nprocesses,
                      [err_criteria] * nprocesses,
                      el_chunks)
    gp_err_list = list(gp_err_list)

    gel_list, pel_list = [{k: v for j in gp_err_list for k, v in j[i].items()}
                          for i in range(2)]

    print("After sanitization, {} good pictures, {} poor pictures".format(len(gel_list), len(pel_list)))
    for fn, errs in [["good_errs.json", gel_list],
                     ["poor_errs.json", pel_list]]:
        with open(fn, "w") as f:
            json.dump(errs, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
