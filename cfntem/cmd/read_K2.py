
import argparse, os, itertools, time, math
from cfntem.io.read_K2 import read_gatan_K2_bin
import numpy as np
from dateutil.relativedelta import relativedelta
import ipyparallel as ipp
import cv2

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
        import os
        from dateutil.relativedelta import relativedelta
    c[:].apply(set_engine_global_variables, gtg_file, frame_duration, out_dir)
    c[:].wait()
    return map_func, len(c.ids)

def convert_image_batch(id_list):
    global datacube, frame_duration, out_dir
    error_messages = []
    for i_frame in id_list:
        try:
            img = datacube.data[i_frame, 0, :, :].mean(axis=0)
            img = (255 * img / img.max()).astype('uint8')
            rt = relativedelta(seconds=frame_duration * i_frame)
            time_txt = f'Hour{int(rt.hours):02d}_Minute{int(rt.minutes):02d}_Second{int(rt.seconds):02d}'
            fn = f'{out_dir}/{time_txt}/{time_txt}_Frame{i_frame % int(1.0/frame_duration)}.png'
            dn = os.path.dirname(fn)
            if not os.path.exists(dn):
                os.makedirs(dn)
            cv2.imwrite(fn, img)
        except Exception as ex:
            error_messages.append(f"Error at frame number {i_frame}:\n{ex}")
    return error_messages


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
    
    start_time = time.time()
    out_dir = os.path.join(args.out_dir, os.path.basename(args.gtg_file).replace("_.gtg", ""))
    map_func, n_procs = get_map_func(args.ipp_dir, args.gtg_file, frame_duration, out_dir)
    print(f"There are {n_frames} frames, will convert using {n_procs}" 
           "processes and allocate {batch_size} images each time")
    err_list = map_func(convert_image_batch, frame_id_batches)
    time_used = time.time() - start_time
    err_list = list(itertools.chain(*err_list))
    if len(err_list) > 0:
        print('\n\n'.join(err_list))
    print(f"Conversion finished. Time used: {time_used:.2f}s.\n\n")



if __name__ == '__main__':
    main()
