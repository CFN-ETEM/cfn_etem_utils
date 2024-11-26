
import argparse, os, itertools, time, math
from datetime import datetime
from cfntem.format_conversion.k2_to_numpy import read_gatan_K2_bin
import numpy as np
from dateutil.relativedelta import relativedelta
import ipyparallel as ipp
import cv2
import logging
from concurrent.futures import ProcessPoolExecutor


def set_engine_global_variables(gtg_file, fm_dur, od):
    global datacube, frame_duration, out_dir, logger
    datacube = read_gatan_K2_bin(gtg_file, mem='MEMMAP', K2_sync_block_IDs=False, K2_hidden_stripe_noise_reduction=False)
    frame_duration = fm_dur
    out_dir = od
    log_dir = f"{os.path.split(out_dir)[0]}/conv_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    engine_id = os.getpid()
    log_path = f"{log_dir}/engine_{engine_id}.txt"
    handler = logging.FileHandler(log_path) # print log in file
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            fmt = '%(asctime)s %(levelname)s:  %(message)s',
            datefmt ='%m-%d %H:%M'
        )
    )
    logger = logging.getLogger(f"engine_{engine_id}_info")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)


def convert_image_batch(id_list):
    global datacube, frame_duration, out_dir, logger
    error_messages = []
    logger.info(f"Started to process {id_list[:3]} to {id_list[-3:]} at {datetime.isoformat(datetime.now())}")
    for j, i_frame in enumerate(id_list):
        try:
            if j < 3:
                logger.info(f"Extract frame {i_frame} at {datetime.isoformat(datetime.now())}")
            img = datacube.data[i_frame, 0, :, :]
            img = (255 * img / img.max()).astype('uint8')
            rt = relativedelta(seconds=frame_duration * i_frame)
            time_txt = f'Hour{int(rt.hours):02d}_Minute{int(rt.minutes):02d}_Second{int(rt.seconds):02d}'
            dir_time_txt = '/'.join(time_txt.split("_"))
            fn = f'{out_dir}/{dir_time_txt}/{time_txt}_Frame{i_frame % int(1.0/frame_duration)}.png'
            dn = os.path.dirname(fn)
            if j < 3:
                logger.info(f"Write frame {i_frame} to PNG at {datetime.isoformat(datetime.now())}")
            if not os.path.exists(dn):
                os.makedirs(dn, exist_ok=True)
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
    parser.add_argument('-b', '--batch_size', type=int, default=10,
                        help='Number of images as an unit for allocation')
    parser.add_argument('--out_dir', type=str, default='ipypar',                     
                        help='The directory to save converted images')
    parser.add_argument('-p', '--processes', type=int, default=8,
                        help='Number of processes run in parallel')
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
    print(f"There are {n_frames} frames, will convert using {args.processes} " 
          f"processes and allocate {batch_size} images each time")
    print("Start conversion", datetime.isoformat(datetime.now()))
    if args.processes == 1:
        set_engine_global_variables(args.gtg_file, frame_duration, out_dir)
        err_list = map(convert_image_batch, frame_id_batches)
    else:
        with ProcessPoolExecutor(args.processes, 
                                 initializer=set_engine_global_variables, 
                                 initargs=((args.gtg_file, frame_duration, out_dir))
                                 ) as executor:
            err_list = executor.map(convert_image_batch, frame_id_batches)
            err_list = list(err_list)
            print("Finished conversion", datetime.isoformat(datetime.now()))
    err_list = list(itertools.chain(*err_list))
    time_used = time.time() - start_time
    if len(err_list) > 0:
        print('\n\n'.join(err_list))
    print(f"Conversion finished. Time used: {time_used:.2f}s.\n\n")



if __name__ == '__main__':
    main()
