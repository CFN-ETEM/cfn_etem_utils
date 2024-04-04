import itertools
import argparse, pathlib, os, json
import hyperspy.api as hs
from PIL import Image
import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="The folder contains original images",
                        required=True)
    parser.add_argument("-t", "--to", help="The folder to hold the average TIFF file",
                        required=True)
    parser.add_argument("-d", "--drift_correction", help="The json file path for drift correction numbers",
                        required=True)
    args = parser.parse_args()
    fn_json = pathlib.Path(args.drift_correction)
    target_dir = pathlib.Path(args.to)
    if target_dir.exists():
        assert target_dir.is_dir()
    else:
        os.makedirs(target_dir, exist_ok=True)
    source_dir = pathlib.Path(args.source)
    ext_list = ['png', 'tiff']
    src_fn_list = list(sorted(itertools.chain(
        *[source_dir.glob(f'*.{ext}') for ext in ext_list])))
    assert len(src_fn_list) > 5
    with open(fn_json) as f:
        d = json.load(f)
    assert len(src_fn_list) - 4  == len(d)
    
    images = [cv2.imread(str(fn), cv2.IMREAD_ANYDEPTH) 
              for fn in src_fn_list]
    images = np.stack(images)
    images = images[2:-2]
    img_shape = images.shape[-2:]
    reference_point = -np.array(d)[:, 0, :].min(axis=0)
    reference_point[reference_point < 0] = 0
    target_cropped_size = np.array(img_shape) \
        -np.array(d)[:, 0, :].max(axis=0) - reference_point
    target_cropped_size = target_cropped_size.astype('int32')
    cropped_images = []
    for i, (img_shift, _) in enumerate(d):
        img_shift = np.array(img_shift)
        img = images[i]
        top_left_corner = (reference_point + img_shift).astype("int32")
        top, left = top_left_corner
        bottom, right = top_left_corner + target_cropped_size
        cropped_img = img[top:bottom, left:right]
        cropped_images.append(cropped_img)
    cropped_images = np.stack(cropped_images)
    drift_corrected_avg_image = cropped_images.mean(axis=0)
    direct_avg_image = images.mean(axis=0)
    
    Image.fromarray(direct_avg_image).save(
        target_dir / f"direct_average_{len(images)}frames.tiff")
    Image.fromarray(drift_corrected_avg_image).save(
        target_dir / f"drift_corrected_average_{len(images)}frames.tiff")