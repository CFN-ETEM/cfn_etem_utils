import itertools
import argparse
import pathlib
from PIL import Image
import numpy as np
import cv2
import sys
import torch
import math

def get_1frame(i_frame, frames):
    assert i_frame >= 2
    assert i_frame <= frames.shape[0] - 3
    img = np.zeros(shape=(1, 5) + frames.shape[1:], dtype=np.float32)
    img[0, ...] = frames[i_frame - 2: i_frame + 3]
    return img

def denoise_image(img, model, device, patch_size):
    n_patch_H = math.ceil(img.shape[-2] / patch_size)
    n_patch_W = math.ceil(img.shape[-1] / patch_size)
    img = torch.tensor(img, device=device, dtype=torch.float32)
    denoised_img = torch.zeros_like(img[0, 0])
    with torch.no_grad():
        for i in range(n_patch_H):
            for j in range(n_patch_W):
                patch = img[:, :, max(i * patch_size - 5, 0): (i+1) * patch_size + 5, max(j * patch_size - 5, 0): (j+1) * patch_size + 5]
                denoised_patch = model(patch)[0]
                denoised_img[i * patch_size: (i+1) * patch_size, j * patch_size: (j+1) * patch_size] = \
                    denoised_patch[0, 0, 5 * min(i, 1):patch_size + 5* min(i, 1), 5*min(j, 1):patch_size + 5* min(j, 1)]
    denoised_img = denoised_img.cpu().detach().numpy().astype(np.float32)
    return denoised_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="The folder contains original images",
                        required=True)
    parser.add_argument("-t", "--to", help="The folder to hold denoised TIFF files",
                        required=True)
    parser.add_argument("--udvd_path", help="The path to the UDVD denoising package",
                        required=True)
    parser.add_argument("-g", "--gpu", help="The index of the GPU to use",
                        default=0, type=int)
    parser.add_argument("-p", "--patch_size", help="The size of patch to split the image",
                        default=550, type=int)

    args = parser.parse_args()
    patch_size = args.patch_size
    udvd_dir = pathlib.Path(args.udvd_path).expanduser()
    sys.path.append(str(udvd_dir))
    import data, utils, models
    target_dir = pathlib.Path(args.to)
    assert target_dir.is_dir()
    source_dir = pathlib.Path(args.source)
    ext_list = ['png', 'tiff']
    src_fn_list = list(sorted(itertools.chain(
        *[source_dir.glob(f'*.{ext}') for ext in ext_list])))
    assert len(src_fn_list) > 5

    device = torch.device(f'cuda:{args.gpu}')
    fn_model = udvd_dir / "pretrained" / "fluoro_micro.pt"
    model, _, _ = utils.load_model(
        fn_model, Fast=False, parallel=True, pretrained=True, old=True, load_opt=False)
    model.to(device)
    model.eval()
    
    noisy_images = [cv2.imread(fn, cv2.IMREAD_ANYDEPTH) 
              for fn in src_fn_list]
    v_high = np.percentile(noisy_images, 99, axis=[1, 2])
    noisy_images = noisy_images.astype(np.float32)
    noisy_images = np.clip(noisy_images, a_min=None, a_max=v_high[:, None, None])
    noisy_images -= noisy_images.min(axis=-1).min(axis=-1)[:, None, None]
    noisy_images /= noisy_images.max(axis=-1).max(axis=-1)[:, None, None]

    for i in range(2, noisy_images.shape[0]-2):
        out_fn = target_dir / f'frame{i+1}.tiff'
        batch = get_1frame(i, noisy_images)
        denoised_image = denoise_image(batch, model, device, patch_size)
        pil_img = Image.fromarray(denoised_image)
        pil_img.save(out_fn)
