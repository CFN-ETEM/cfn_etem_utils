import os, sys

import cv2
import hyperspy.api as hs
from PIL import Image
import numpy as np
from ict.particle_tracking.image_processing import ImageEnhance, ImageMerge, ImageAverage, ImageMaskShrinkAndShift
from ict.particle_tracking.utils import estimate_detection_error


def load_dm4_file(fn, processors=(), no_minimum_subtraction=False, normalize_intensity=-1):
    if os.path.splitext(fn)[1] in [".dm4", ".dm3"]:
        img = hs.load(fn).data
    elif os.path.splitext(fn)[1] in [".tiff"]:
        img_pil = Image.open(fn)
        img = np.array(img_pil)
    else:
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    img = ImageEnhance.whiten(img, no_minimum_subtraction=no_minimum_subtraction, 
                              normalize_intensity=normalize_intensity)
    for p in processors:
        img = p.process(img)
    return img

def binarize_file_list(in_file_list, focus_index, processors,
                       out_file=None, record_file=None, estimate_error=False, marker_thickness=1,
                       merge_only=False, raw_file=None):
    non_merge_processors = [p for p in processors if not isinstance(p, (ImageMerge, ImageMaskShrinkAndShift))]
    merge_processor = [p for p in processors if isinstance(p, ImageMerge)][0]
    fix_processors = [p for p in processors if isinstance(p, ImageMaskShrinkAndShift)]

    window_size = merge_processor.params.get("window_size", 1)
    if not merge_only:
        images_unmerged = [load_dm4_file(fn, non_merge_processors)
            for fn in ImageMerge.get_sub_window(in_file_list, focus_index, window_size)]
    else:
        images_unmerged = [cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
                           for fn in ImageMerge.get_sub_window(in_file_list, focus_index, window_size)]
    if window_size > 1:
        img_merged = merge_processor.process(images_unmerged)
    else:
        img_merged = images_unmerged[0]
        assert estimate_error is False

    for p in fix_processors:
        img_merged = p.process(img_merged)
        
    if estimate_error:
        if window_size > 1 and merge_processor.last_err is not None:
            trust_ratio = merge_processor.last_err
        else:
            trust_ratio = estimate_detection_error(img_merged, images_unmerged)
    else:
        trust_ratio = None
    if out_file is not None:
        out_dir = os.path.dirname(out_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(out_file, img_merged)
    img_marked = None
    if record_file is not None:
        if raw_file is not None:
            img_orig = load_dm4_file(raw_file)
        else:
            img_orig = load_dm4_file(in_file_list[focus_index])
        contours, _ = cv2.findContours(img_merged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_marked = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(img_marked, contours, contourIdx=-1, color=[0, 255, 0], thickness=marker_thickness)
        rec_dir = os.path.dirname(record_file)
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir)
        cv2.imwrite(record_file, img_marked, [cv2.IMWRITE_PNG_COMPRESSION, 8])
    return img_merged, img_marked, trust_ratio


def average_file_list(in_file_list, focus_index, window_size, out_file=None):
    avg_processor = ImageAverage()
    src_fn_list = ImageMerge.get_sub_window(in_file_list, focus_index, window_size)
    images_orig = [load_dm4_file(fn)
                   for fn in src_fn_list]
    img_avg = None
    err = None
    try:
        img_avg = avg_processor.process(images_orig)
        if out_file is not None:
            out_dir = os.path.dirname(out_file)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            cv2.imwrite(out_file, img_avg)
    except:
        err = str(sys.exc_info())
    err_info = None
    if err is not None:
        err_info = dict(message=err, offending_files=src_fn_list)
    return img_avg, err_info