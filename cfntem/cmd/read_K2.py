
import argparse, os, itertools, time, math
from datetime import datetime
from cfntem.io.read_K2 import read_gatan_K2_bin
import numpy as np
from dateutil.relativedelta import relativedelta
import ipyparallel as ipp
import cv2
import logging

def set_engine_global_variables(gtg_file, fm_dur, od):
    global datacube, frame_duration, out_dir, engine_id
    global logger
    dn = os.path.dirname(gtg_file)
    bn = os.path.basename(gtg_file)
    proc_gtg_fn = f"{dn}/copies/copy_{engine_id+1}/{bn}"
    datacube = read_gatan_K2_bin(proc_gtg_fn, mem='MEMMAP', K2_sync_block_IDs=False, K2_hidden_stripe_noise_reduction=False)
    frame_duration = fm_dur
    out_dir = od
    log_dir = f"{out_dir.split('out_images')[0][:-1]}/conv_logs"
    log_path = f"{log_dir}/engine_{engine_id:02d}.txt"
    handler = logging.FileHandler(log_path) # print log in file
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            fmt = '%(asctime)s %(levelname)s:  %(message)s',
            datefmt ='%m-%d %H:%M'
        )
    )
    logger = logging.getLogger(f"engine_{engine_id:02d}_info")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

def set_engine_id(ei):
    global engine_id
    engine_id = ei

def get_map_func(ipp_dir, gtg_file, frame_duration, out_dir):
    c = ipp.Client(
        connection_info=f"{ipp_dir}/security/ipcontroller-client.json"
    )
    map_func =  c[:].map_sync
    with c[:].sync_imports():
        from cfntem.io.read_K2 import read_gatan_K2_bin
        import cv2
        import os, logging
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
    for i, en in enumerate(c):
        en.apply(set_engine_id, i)
    c[:].apply(set_engine_global_variables, gtg_file, frame_duration, out_dir)
    c[:].wait()
    return map_func, len(c.ids)

def convert_image_batch(id_list):
    global datacube, frame_duration, out_dir, engine_id
    logger = logging.getLogger(f"engine_{engine_id:02d}_info")
    error_messages = []
    logger.info(f"Started to process {id_list[:3]} to {id_list[-3:]} at {datetime.isoformat(datetime.now())}")
    for j, i_frame in enumerate(id_list):
        try:
            if j < 3:
                logger.info(f"Extract frame {i_frame} at {datetime.isoformat(datetime.now())}")
            img = datacube.data[i_frame, 0, :, :].mean(axis=0)
            img = (255 * img / img.max()).astype('uint8')
            rt = relativedelta(seconds=frame_duration * i_frame)
            time_txt = f'Hour{int(rt.hours):02d}_Minute{int(rt.minutes):02d}_Second{int(rt.seconds):02d}'
            dir_time_txt = '/'.join(time_txt.split("_")[:-1])
            fn = f'{out_dir}/{dir_time_txt}/{time_txt}_Frame{i_frame % int(1.0/frame_duration)}.png'
            if j < 3:
                logger.info(f"Write frame {i_frame} to PNG at {datetime.isoformat(datetime.now())}")
            cv2.imwrite(fn, img)
            if j < 3:
                logger.info(f"Processed frame {i_frame} at {datetime.isoformat(datetime.now())}")
        except Exception as ex:
            error_messages.append(f"Error at frame number {i_frame}:\n{ex}")
    logger.info(f"Finished processing {id_list[:3]} to {id_list[-3:]} at {datetime.isoformat(datetime.now())}\n\n")
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
    parser.add_argument('-s', '--sequential', action='store_true',
                        help='Run the conversion sequentially')
    args = parser.parse_args()
    print("first round reading", datetime.isoformat(datetime.now()))
    datacube = read_gatan_K2_bin(args.gtg_file, mem='MEMMAP', K2_sync_block_IDs=False, K2_hidden_stripe_noise_reduction=False)
    n_frames = datacube.data.shape[0]
    frame_duration = datacube.data._gtg_meta['.Acquisition.Frame.Sequence.Frame Exposure (s)']
    batch_size = int(args.batch_size)
    frame_id_batches = np.arange(math.ceil(n_frames/batch_size)*batch_size).reshape(-1, batch_size).tolist()
    frame_id_batches[-1] = frame_id_batches[-1][:n_frames%batch_size]
    
    start_time = time.time()
    out_dir = os.path.join(args.out_dir, os.path.basename(args.gtg_file).replace("_.gtg", ""))
    print("Set up parallel or sequential engine", datetime.isoformat(datetime.now()))
    log_dir = f"{out_dir.split('out_images')[0][:-1]}/conv_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if args.sequential:
        map_func, n_procs = map, 1
        set_engine_id(0)
        set_engine_global_variables(args.gtg_file, frame_duration, out_dir)
    else:
        map_func, n_procs = get_map_func(args.ipp_dir, args.gtg_file, frame_duration, out_dir)
    print(f"There are {n_frames} frames, will convert using {n_procs} " 
          f"processes and allocate {batch_size} images each time")
    print("Start conversion", datetime.isoformat(datetime.now()))
    err_list = []
    created_dn_set = set()
    for i_seg in range(math.ceil(len(frame_id_batches)/batch_size)):
        seg_batches = frame_id_batches[i_seg*batch_size: (i_seg+1)*batch_size]
        print(f"Process segment {seg_batches[0][:3]} to {seg_batches[-1][-3:]} at {datetime.isoformat(datetime.now())}")
        seg_id_list = list(itertools.chain(*seg_batches))
        seg_dn_list = []
        for i_frame in seg_id_list:
            rt = relativedelta(seconds=frame_duration * i_frame)
            time_txt = f'Hour{int(rt.hours):02d}_Minute{int(rt.minutes):02d}_Second{int(rt.seconds):02d}'
            dir_time_txt = '/'.join(time_txt.split("_")[:-1])
            dn = f'{out_dir}/{dir_time_txt}'
            seg_dn_list.append(dn)
        seg_dn_list = set(seg_dn_list)
        new_dn_list = list(seg_dn_list - created_dn_set)
        for dn in new_dn_list:
            os.makedirs(dn)
        created_dn_set = created_dn_set + new_dn_list
        el = map_func(convert_image_batch, seg_batches)
        err_list.extend(el)
    print("Finished conversion", datetime.isoformat(datetime.now()))
    err_list = list(itertools.chain(*err_list))
    time_used = time.time() - start_time
    if len(err_list) > 0:
        print('\n\n'.join(err_list))
    print(f"Conversion finished. Time used: {time_used:.2f}s.\n\n")



if __name__ == '__main__':
    main()
