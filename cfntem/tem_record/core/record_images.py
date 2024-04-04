import time
from PIL import Image
from datetime import datetime
import os

from gatansocket import GatanSocket


class Args():
    def __init__(self):
        self.out_dir = "/Users/xiaohuiqu/Downloads/tem_images"
        self.interval = 3
        
        self.processing = 'gain normalized'
        self.height = 2672
        self.width = 4008
        self.binning = 1
        self.top = 0
        self.left = 0 
        self.bottom = 2672
        self.right = 4008
        self.exposure = 3
        self.shutterDelay = 0


class PeriodicRecorder:
    def __init__(self, img_dir):
        self.remote_gatan = GatanSocket()
        self.img_dir = os.path.abspath(img_dir)
        self.t0 = datetime.now()
        

    def compose_file_name(self):
        cur_t = datetime.now()
        rel_t = cur_t - self.t0
        hour_text, min_text, sec_text =str(rel_t).split(".")[0].split(':')
        if len(hour_text) == 1:
            hour_text = '0' + hour_text
        fn = 'Hour{}_Minute{}_Second{}.tiff'.format(hour_text, min_text, sec_text)
        dn = "Hour" + hour_text
        return dn, fn

    def save_image(self, record_args):
        dn, fn = self.compose_file_name()
        start_t = datetime.now()
        np_img = self.remote_gatan.GetImage(
            record_args.processing, record_args.height, record_args.width, 
            record_args.binning, record_args.top, record_args.left, record_args.bottom, 
            record_args.right, record_args.exposure, record_args.shutterDelay)
        end_t = datetime.now()
        rel_t = end_t - start_t
        retrieval_seconds = rel_t.total_seconds()
        retrieval_time_text = "_Record{:.1f}s".format(retrieval_seconds)
        fn_base, fn_ext = os.path.splitext(fn)
        fn = fn_base + retrieval_time_text + fn_ext
        pil_img = Image.fromarray(np_img, 'I;16')
        print("write file {}".format(fn))
        dn = os.path.join(self.img_dir)
        fn = os.path.join(dn, fn)
        if not os.path.exists(dn):
            os.makedirs(dn)
        pil_img.save(fn)


def main():
    args = Args()
    pr = PeriodicRecorder(args.out_dir)

    while True:
        start_t = datetime.now()
        pr.save_image(args)
        end_t = datetime.now()
        rel_t = end_t - start_t
        retrieval_seconds = rel_t.total_seconds()
        assert args.interval > retrieval_seconds
        time.sleep(args.interval - retrieval_seconds)
        

if __name__ == '__main__':
    main()
