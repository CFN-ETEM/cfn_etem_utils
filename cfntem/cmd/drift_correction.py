import argparse
import copy
import cv2
import glob
import json
import math
import os
import pathlib
import re
import socket
import subprocess
from itertools import chain

import ipyparallel as ipp
import numpy

from cfntem.particle_tracking.image_processing import ImageDriftCorrection
from cfntem.particle_tracking.io import load_dm4_file


def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        return arg


def init_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="The folder constains binary image",
                        required=True)
    parser.add_argument("-t", "--to", help="The folder to hold the cropped image",
                        required=True)
    parser.add_argument("--raw_in", help="The folder contain the raw image before binarization",
                        required=False)
    parser.add_argument("--raw_out", help="The folder to put cropped raw image",
                        required=False)
    parser.add_argument("--raw_format", help="The file format of raw image",
                        default="dm4",
                        type=str)
    parser.add_argument("--mark_interval", help="The interval to mark a raw image",
                        default=10,
                        type=int)
    parser.add_argument("-w", "--target_cropped_size", help="The size of target cropped image",
                        default=1200,
                        type=int)
    parser.add_argument("-m", "--maxshift", help="maximum shift in pixel between consecutive frames",
                        default=50,
                        type=int)
    parser.add_argument("-r", "--rebase_shift",
                        help="Do a reference point rebase when accumulated shift exceeds this value",
                        default=10,
                        type=int)
    parser.add_argument("--rebase_steps", help="Rebase interval in steps",
                        default=2000,
                        type=int)
    parser.add_argument("-d", "--drift_save", help="Save the drift correction list",
                        action='store_true')
    parser.add_argument("-p", "--parallel", help="Run the job in parallel mode",
                        action='store_true')
    parser.add_argument("--fps", help="Frame rate to write video, negative value mean no video writting", default=-1, type=int)
    parser.add_argument("-nms", "--no_minimum_subtraction", help="Don't subtract "
                        "the minimum in image whitenning",
                        action='store_true')
    parser.add_argument("-ni", "--normalize_intensity", help="Normalize the average "
                        "intensity to an averaged value, in the range of 0~255. "
                        "Negative value will just stretch the intensity to full "
                        "dynamic range of the integer",
                        default=-1,
                        type=int)
    args = parser.parse_args()
    source_dir = pathlib.Path(args.source).expanduser().absolute()
    assert source_dir.exists()
    assert source_dir.is_dir()
    source_dir = str(source_dir)
    dest_dir = pathlib.Path(args.to).expanduser().absolute()
    if not dest_dir.exists():
        os.makedirs(dest_dir, exist_ok=True)
    else:
        assert dest_dir.is_dir()
    dest_dir = str(dest_dir)
    raw_dir_in = args.raw_in
    raw_dir_out = args.raw_out
    raw_format = args.raw_format
    mark_interval = args.mark_interval
    target_cropped_size = args.target_cropped_size
    maxshift = args.maxshift
    rebase_shift = args.rebase_shift
    rebase_steps = args.rebase_steps
    drift_save = args.drift_save
    parallel_run = args.parallel
    fps = args.fps
    no_minimum_subtraction = args.no_minimum_subtraction
    normalize_intensity = args.normalize_intensity
    in_file_list = list(chain(*[glob.glob(f"{source_dir}/Capture{'[0-9]'*n}/**/*.{fmt}", recursive=True)
                                for n in range(1, 4)
                                for fmt in ["png", "dm4"]]))
    if len(in_file_list) == 0:
        in_file_list = glob.glob("{}/**/*.png".format(source_dir), recursive=True)
    if len(in_file_list) == 0:
        in_file_list = glob.glob("{}/*.tiff".format(source_dir), recursive=True)
    if len(in_file_list) == 0:
        print("Error: no files found")
        exit(1)
    if re.search(r'Capture(\d+).*', in_file_list[0]) is not None:
        in_file_list = sorted(in_file_list, key=lambda x: (
            int(re.search(r'Capture(\d+).*', x).group(1)),
            x))
    else:
        in_file_list = sorted(in_file_list)

    return parallel_run, source_dir, dest_dir, raw_dir_in, raw_dir_out, raw_format, mark_interval, target_cropped_size, \
           maxshift, rebase_shift, rebase_steps, drift_save, fps, in_file_list, no_minimum_subtraction, \
           normalize_intensity


def get_parallel_map_func():
    c = ipp.Client(connection_info="ipypar/security/ipcontroller-client.json")
    print("Engine IDs:", c.ids)
    with c[:].sync_imports():
        import cv2, os, numpy, json, socket
        from cfntem.particle_tracking.image_processing import ImageDriftCorrection
        from cfntem.particle_tracking.io import load_dm4_file
    c[:].push(dict(crop_image=crop_image),
              block=True)

    return c[:].map_sync, len(c.ids)


def find_corrections(img_shape, target_cropped_size, maxshift, rebase_shift, rebase_steps, in_file_list, 
                     no_minimum_subtraction, normalize_intensity):
    dc_processor = ImageDriftCorrection(img_shape, target_cropped_size, maxshift, rebase_shift, rebase_steps)
    for fn in in_file_list:
        img = load_dm4_file(fn, no_minimum_subtraction=no_minimum_subtraction, normalize_intensity=normalize_intensity)
        dc_processor.process(img)
    with open("checkpoints/drift_corrections.pid{}.{}.json".format(os.getpid(), socket.gethostname()), "w") as f:
        json.dump(dc_processor.corrections, f, indent=4)
    return dc_processor.corrections


def write_cropped_files(in_file_list, corrections, source_dir, dest_dir, raw_dir_in, raw_dir_out, raw_format,
                        mark_interval, target_cropped_size, reference_point, group_number, fps,
                        no_minimum_subtraction, normalize_intensity):
    assert len(in_file_list) == len(corrections)
    num_no_mark_files = 0
    png_file_list = []
    for i, (fn, corr) in enumerate(zip(in_file_list, corrections)):
        out_fn = os.path.abspath(fn).replace(source_dir, dest_dir)
        if out_fn is not None:
            if "dm4" in os.path.splitext(out_fn)[-1]:
                out_fn = out_fn.replace(".dm4", ".png")
            if "tiff" in os.path.splitext(out_fn)[-1]:
                out_fn = out_fn.replace(".tiff", ".png")
            out_dir = os.path.dirname(out_fn)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            img = load_dm4_file(fn, no_minimum_subtraction=no_minimum_subtraction, normalize_intensity=normalize_intensity)
            if img.size < numpy.prod(target_cropped_size) or img.max() < 120:
                continue
            cropped_img = crop_image(corr, img, target_cropped_size, reference_point)
            cv2.imwrite(out_fn, cropped_img)
            png_file_list.append(f"file {out_fn} \n")
            if fps > 0:
                png_file_list.append(f"duration {1.0/fps} \n")
        if num_no_mark_files % mark_interval == 0 and raw_dir_in is not None and raw_dir_out is not None:
            raw_fn_in = os.path.abspath(fn).replace(source_dir, raw_dir_in).replace("png", raw_format)
            cropped_raw_fn = os.path.abspath(fn).replace(source_dir, raw_dir_out)
            if os.path.exists(raw_fn_in):
                img_raw_in = load_dm4_file(raw_fn_in, no_minimum_subtraction=no_minimum_subtraction, normalize_intensity=normalize_intensity)
                img_raw_cropped = crop_image(corr, img_raw_in, target_cropped_size, reference_point)
                raw_out_dir = os.path.dirname(cropped_raw_fn)
                if not os.path.exists(raw_out_dir):
                    os.makedirs(raw_out_dir, exist_ok=True)
                cv2.imwrite(cropped_raw_fn, img_raw_cropped, [cv2.IMWRITE_PNG_COMPRESSION, 8])
                num_no_mark_files = -1
        num_no_mark_files += 1
    if fps > 0:
        clip_image_fn = f"video_clips/clip_image_list_{group_number}.txt"
        with open(clip_image_fn, "w") as f:
            f.writelines(png_file_list)
        ffmpeg_cmd = f"ffmpeg  -f concat -safe 0 -i {clip_image_fn} -r {fps} -pix_fmt yuv420p" \
                    f" -vcodec libx264 -crf 30 video_clips/clip_{group_number}.mp4"
        subprocess.Popen(ffmpeg_cmd, shell=True).wait()


def crop_image(corr, img, target_cropped_size, reference_point):
    assert len(corr) == 2
    assert len(corr[0]) == 2
    img_shift = numpy.array(corr[0])
    assert img_shift.shape == (2,)
    assert target_cropped_size.shape == (2,)
    assert reference_point.shape == (2, )
    top_left_corner = (reference_point + img_shift).astype("int32")
    top, left = top_left_corner
    bottom, right = top_left_corner + target_cropped_size
    cropped_img = img[top:bottom, left:right]
    return cropped_img


def main():
    parallel_run, source_dir, dest_dir, raw_dir_in, raw_dir_out, raw_format, mark_interval, target_cropped_size, \
    maxshift, rebase_shift, rebase_steps, drift_save, fps, full_in_file_list, no_minimum_subtraction, \
    normalize_intensity = init_params()
    if parallel_run:
        par_map, nprocesses = get_parallel_map_func()
    else:
        par_map, nprocesses = map, 1
    print("running with {} processes".format(nprocesses))

    example_img = load_dm4_file(full_in_file_list[0], no_minimum_subtraction=no_minimum_subtraction, normalize_intensity=normalize_intensity)
    img_shape = example_img.shape
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    nfiles_per_process = math.ceil(len(full_in_file_list) / nprocesses)
    ifl_chunks = [full_in_file_list[i * nfiles_per_process: (i + 1) * nfiles_per_process]
                  for i in range(nprocesses)]
    orig_ifl_chunck = copy.deepcopy(ifl_chunks)
    for i in range(nprocesses - 1):
        if len(ifl_chunks[i + 1]) > 0:
            ifl_chunks[i].append(ifl_chunks[i + 1][0])
        else:
            break
    corr_list = par_map(find_corrections,
                    [img_shape] * nprocesses,
                    [target_cropped_size] * nprocesses,
                    [maxshift] * nprocesses,
                    [rebase_shift] * nprocesses,
                    [rebase_steps] * nprocesses,
                    ifl_chunks,
                    [no_minimum_subtraction] * nprocesses, 
                    [normalize_intensity] * nprocesses)
    corr_list = list(corr_list)
    if drift_save:
        with open("corr_list_before_merge.json", "w") as f:
            json.dump(corr_list, f, indent=4)
    corrections = []
    last_drift = [0, 0]
    for i, corr in enumerate(corr_list):
        if i == 0:
            cc = corr

        else:
            cc = [[[p + prev_p for p, prev_p in zip(img_shift, last_drift)], step_shift]
                  for img_shift, step_shift in corr]
        if len(cc) > 0:
            last_drift = cc[-1][0]
        if i < len(corr_list) - 1:
            cc = cc[:-1]
        corrections.append(cc)
    corrections_one = list(chain(*corrections))
    if drift_save:
        with open("drift_corrections.json", "w") as f:
            json.dump(corrections_one, f, indent=4)

    safe_margin = 2
    reference_point = -numpy.array(corrections_one)[:, 0, :].min(axis=0)
    reference_point[reference_point < 0] = 0
    reference_point += safe_margin
    adjusted_target_cropped_size = numpy.array(img_shape) \
                                   - numpy.array(corrections_one)[:, 0, :].max(axis=0) \
                                   - safe_margin \
                                   - reference_point
    adjusted_target_cropped_size = adjusted_target_cropped_size.astype('int32')
    adjusted_target_cropped_size = (adjusted_target_cropped_size // 2) * 2
    print("Adjust target window size to {}".format(adjusted_target_cropped_size))
    if fps > 0 and not os.path.exists("video_clips"):
        os.makedirs("video_clips")
    list(
        par_map(write_cropped_files,
            orig_ifl_chunck, corrections,
            [source_dir] * nprocesses,
            [dest_dir] * nprocesses,
            [raw_dir_in] * nprocesses,
            [raw_dir_out] * nprocesses,
            [raw_format] * nprocesses,
            [mark_interval] * nprocesses,
            [adjusted_target_cropped_size] * nprocesses,
            [reference_point] * nprocesses,
            list(range(nprocesses)),
            [fps] * nprocesses,
            [no_minimum_subtraction] * nprocesses, 
            [normalize_intensity] * nprocesses)
    )
    
    if fps > 0:
        clip_fn_list = [f"file clip_{i}.mp4 \n" for i in range(nprocesses)]
        with open("video_clips/clip_fn_list.txt", "w") as f:
            f.writelines(clip_fn_list)

        ffmpeg_cmd = "ffmpeg -y -loglevel error -f concat -safe 0 -i video_clips/clip_fn_list.txt" \
                    " -vcodec copy cropped_video.mp4"
        subprocess.Popen(ffmpeg_cmd, shell=True).wait()

    assert len(corrections_one) == len(full_in_file_list)

if __name__ == '__main__':
    main()

