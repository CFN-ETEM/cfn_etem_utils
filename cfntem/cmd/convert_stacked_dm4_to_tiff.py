    
import argparse, pathlib, os
import hyperspy.api as hs
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="The stacked DM4 file name",
                        required=True)
    parser.add_argument("-t", "--to", help="The folder to hold the TIFF files",
                        required=True)
    args = parser.parse_args()
    fn_dm4 = pathlib.Path(args.source)
    images = hs.load(fn_dm4).data
    dir_tiff = pathlib.Path(args.to)
    if not dir_tiff.exists():
        os.makedirs(dir_tiff, exist_ok=True)
    for j, img in enumerate(images):
        Image.fromarray(img).save(dir_tiff / f"frame_{j:03d}.tiff")

