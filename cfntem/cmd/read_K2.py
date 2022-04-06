
import argparse
from cfntem.io.read_K2 import read_gatan_K2_bin
import numpy as np
import math
from dateutil.relativedelta import relativedelta
import ipyparallel as ipp
import cv2, os

def set_engine_global_variables(gtg_file, fm_dur, od):
    global datacube, frame_duration, out_dir
    datacube = read_gatan_K2_bin(gtg_file, mem='MEMMAP', K2_sync_block_IDs=False, K2_hidden_stripe_noise_reduction=False)
    frame_duration = fm_dur
    out_dir = od

def get_map_func(ipp_dir, gtg_file, frame_duration, out_dir):
    c = ipp.Client(
        connection_info=f"{ipp_dir}/security/ipcontroller-client.json"
    )
    map_func =  c.load_balanced_view().map_sync
    with c[:].sync_imports():
        from cfntem.io.read_K2 import read_gatan_K2_bin
        import cv2
        from dateutil.relativedelta import relativedelta
    c[:].apply(set_engine_global_variables, gtg_file, frame_duration, out_dir)
    c[:].wait()
    return map_func, len(c.ids)

def convert_image_batch(id_list):
    global datacube, frame_duration, out_dir
    for i_frame in id_list:
        img = datacube.data[i_frame, 0, :, :].mean(axis=0)
        img = (255 * img / img.max()).astype('uint8')
        rt = relativedelta(seconds=frame_duration * i_frame)
        time_txt = f'Hour{int(rt.hous):02d}_Minute{int(rt.minutes):02d}_Second{int(rt.seconds)}'
        fn = f'{out_dir}/{time_txt}/{time_txt}_Frame{i_frame % int(1.0/frame_duration)}.png'
        cv2.imwrite(fn, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gtg_file', type=str, required=True,
                        help='The path of GTG file')
    parser.add_argument('-b', '--batch_size', type=str, required=True,
                        help='Number of images as an unit for allocation')
    parser.add_argument('--ipp_dir', type=str, default='ipypar',
                        help='The directory for IpyParallel environment')
    parser.add_argument('--out_dir', type=str, default='ipypar',
                        help='The directory to save converted images')
    args = parser.parse_args()
    datacube = read_gatan_K2_bin(args.gtg_file, mem='MEMMAP', K2_sync_block_IDs=False, K2_hidden_stripe_noise_reduction=False)
    n_frames = datacube.data.shape[0]
    frame_duration = datacube.data._gtg_meta['.Acquisition.Frame.Sequence.Frame Exposure (s)']
    batch_size = int(args.batch_size)
    frame_id_batches = np.arange(math.ceil(n_frames/batch_size)*batch_size).reshape(-1, batch_size).tolist()
    frame_id_batches[-1] = frame_id_batches[-1][:n_frames%batch_size]
    
    out_dir = os.path.join(args.out_dir, os.path.basename(args.gtg_file).replace("*_.gtg", ""))
    map_func, n_procs = get_map_func(args.ipp_dir, args.gtg_file, frame_duration, out_dir)
    print(f"There are {n_frames} frames, will convert using {n_procs}" 
           "processes and allocate {batch_size} images each time")
    map_func(frame_id_batches)



if __name__ == '__main__':
    main()
