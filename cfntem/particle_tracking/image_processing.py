from inspect import signature

import numpy as np
import cv2, os, json, abc, copy
from cucim.skimage.registration import phase_cross_correlation as gpu_phase_cross_correlation
from skimage.registration import phase_cross_correlation as cpu_phase_cross_correlation 
import cupy

module_dir = os.path.abspath(os.path.dirname(__file__))


class ImageProcessor(object):

    def __init__(self):
        pass

    def load_params(self, param_file):
        with open(param_file, "r") as f:
            self.params = json.load(f)
        return self.params

    @abc.abstractmethod
    def process(self, img):
        pass


class ImageEnhance(ImageProcessor):

    def __init__(self, param_file=None, include_whiten=False):
        if param_file is None:
            param_file = os.path.join(module_dir, "data", "image_enhance_params.json")
        self.params = self.load_params(param_file)
        self.include_whiten = include_whiten
        super().__init__()

    @staticmethod
    def whiten(img, no_minimum_subtraction=False, normalize_intensity=-1):
        m_val = img.min()
        if no_minimum_subtraction:
            m_val = 0
        if normalize_intensity <= 0:
            alpha=np.iinfo(np.uint8).max / (img.max() - m_val)
        else:
            alpha=normalize_intensity / (img - m_val).mean()
        img = cv2.convertScaleAbs(img - m_val,
            alpha=alpha)
        return img

    @staticmethod
    def _smooth_and_histogram_equalization(img, reverse_color=False, pre_bilat=True,
        bilat_r_space=25, bilat_r_color=50, use_hist_eqz=True, hist_clip_limit=3.0,
        hist_grid_size=8, post_bilat=False):
        if reverse_color:
            img = np.iinfo(img.dtype).max - img
        if pre_bilat:
            img = cv2.bilateralFilter(img, d=-1, sigmaColor=bilat_r_color, sigmaSpace=bilat_r_space)
        if use_hist_eqz:
            clahe = cv2.createCLAHE(clipLimit=hist_clip_limit, tileGridSize=(hist_grid_size, hist_grid_size))
            img = clahe.apply(img)
        if post_bilat:
            img = cv2.bilateralFilter(img, d=-1, sigmaColor=bilat_r_color, sigmaSpace=bilat_r_space)
        return img

    def process(self, img):
        if self.include_whiten:
            img2 = self.whiten(img)
        else:
            img2 = img
        img3 = self._smooth_and_histogram_equalization(img2, **self.params)
        return img3


class ImageBinarization(ImageProcessor):

    def __init__(self, param_file=None):
        if param_file is None:
            param_file = os.path.join(module_dir, "data", "binarization_params.json")
        self.params = self.load_params(param_file)
        super().__init__()

    def _fill_edges(self, img):
        img_inv = cv2.bitwise_not(img)
        # open a fill path from the boarder, avoid a single partile at the corner stops fill
        img_inv[:, :1] = 255
        img_inv[:, -1:] = 255
        img_inv[:1, :] = 255
        img_inv[-1:, :] = 255
        img_fill = img_inv.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = img_fill.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # Floodfill from point (0, 0)
        cv2.floodFill(img_fill, mask, (0, 0), 0)
        return img_fill

    def _exclude_edge_with_particles(self, edges, particles, exclude_kernel_size,
                                     exclude_iter):
        kernel = np.ones((exclude_kernel_size, exclude_kernel_size), np.uint8)
        expanded_particles = cv2.morphologyEx(particles, cv2.MORPH_DILATE, kernel, iterations=exclude_iter)
        mask = cv2.bitwise_not(expanded_particles)
        survive_edges = cv2.bitwise_and(edges, mask)
        return survive_edges

    def _find_particles(self, img, edge_thr, use_l2_gradient,
            edge_fix_kernel_size, sobel_size, edge_fix_iter, denoise_kernel_size, denoise_iter,
            exclude_kernel_size, exclude_iter, expansion_kernel_size, expansion_iter, known_particles):
        edge_thr_weak, edge_thr_sure = edge_thr
        edges = cv2.Canny(img, edge_thr_weak, edge_thr_sure, apertureSize=sobel_size, L2gradient=use_l2_gradient)
        if known_particles is not None:
            edges = self._exclude_edge_with_particles(edges=edges, exclude_kernel_size=exclude_kernel_size,
                                                exclude_iter=exclude_iter, particles=known_particles)
        edge_fix_kernel = np.ones((edge_fix_kernel_size, edge_fix_kernel_size), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_fix_kernel, iterations=edge_fix_iter)
        particles = self._fill_edges(closed_edges)
        denoise_kernel = np.ones((denoise_kernel_size, denoise_kernel_size), np.uint8)
        denoised_particles = cv2.morphologyEx(particles, cv2.MORPH_OPEN, denoise_kernel, iterations=denoise_iter)
        if expansion_kernel_size > 0 and expansion_iter > 0:
            expansion_kernel = np.ones((expansion_kernel_size, expansion_kernel_size), np.uint8)
            expanded_particles = cv2.morphologyEx(denoised_particles, cv2.MORPH_DILATE, expansion_kernel,
                                                  iterations=expansion_iter)
        else:
            expanded_particles = denoised_particles
        return expanded_particles

    def _recursive_find_particles(self, img, **kwargs):
        """

        :param img:
        :param kwargs: Each key with prefix a1_, a2_, a3_
        :return:
        """
        sub_sig = signature(self._find_particles)
        sub_key_list = list(sub_sig.parameters.keys())
        sub_key_list.remove('img')
        sub_key_list.remove('known_particles')
        full_key_list = ['a{}_{}'.format(attempt, k) for attempt in range(1, 4) for k in sub_key_list]
        assert set(kwargs.keys()) == set(full_key_list)

        known_particles = None
        detections = []
        img_orig = img.copy()
        for attempt in range(1, 4):
            sub_kwargs_keys = ['a{}_{}'.format(attempt, k) for k in sub_key_list]
            sub_kwargs = {fk: kwargs[wk] for fk, wk in zip(sub_key_list, sub_kwargs_keys)}
            kp = self._find_particles(img=img_orig, **sub_kwargs,
                                known_particles=known_particles)
            detections.append(kp)
            if known_particles is not None:
                known_particles = cv2.bitwise_or(known_particles, kp)
            else:
                known_particles = kp.copy()
        return known_particles, detections

    def get_process_hiechary(self, img):
        return self._recursive_find_particles(img, **self.params)

    def process(self, img):
        particles, _ = self.get_process_hiechary(img)
        return particles


class ImageMerge(ImageProcessor):

    def __init__(self, param_file=None):
        if param_file is None:
            param_file = os.path.join(module_dir, "data", "frame_merge_params.json")
        self.params = self.load_params(param_file)
        self.last_err = None
        super().__init__()

    def _get_merged_binary_at(self, images, window_size, trust_detection_rate, worse_image_removal_ratio=None,
                              improvement_threshold_to_remove_worst=None, morphology_close_kernel_size=-1,
                              morphology_close_kernel_iter=-1):
        """

        :param images: List of binarized images in the window.
        :param index:
        :param window_size:
        :param trust_detection_rate:
        :return:
        """
        assert len(images) == window_size
        avg_img = np.stack(images, axis=-1).mean(axis=-1).astype('uint8')
        threshold = round(255 * trust_detection_rate)
        _, merged_bin = cv2.threshold(avg_img, threshold, 255, cv2.THRESH_BINARY)
        n_worst = round(window_size * worse_image_removal_ratio) if worse_image_removal_ratio is not None else 0
        self.last_err = None
        if n_worst > 0:
            from cfntem.particle_tracking.utils import estimate_detection_error
            diff_list = []
            err1 = estimate_detection_error(merged_bin, images)
            self.last_err = copy.deepcopy(err1)
            for img in images:
                diff_list.append(cv2.bitwise_xor(merged_bin, img).sum())
            diff_list = np.array(diff_list)
            n_worst = 2
            diff_cutoff = sorted(diff_list, reverse=True)[n_worst]
            truncated_images = [images[i] for i in np.where(diff_list > diff_cutoff)[0]]
            truncated_avg_img = np.stack(truncated_images, axis=-1).mean(axis=-1).astype('uint8')
            _, truncated_merged_bin = cv2.threshold(truncated_avg_img, threshold, 255, cv2.THRESH_BINARY)
            err2 = estimate_detection_error(truncated_merged_bin, truncated_images)
            max_err1 = np.fabs(np.array([err1[k] for k in sorted(err1.keys())])).max()
            max_err2 = np.fabs(np.array([err2[k] for k in sorted(err1.keys())])).max()
            if max_err2 / max_err1 < improvement_threshold_to_remove_worst:
                self.last_err = copy.deepcopy(err2)
                merged_bin = truncated_merged_bin
        if morphology_close_kernel_size > 0 and morphology_close_kernel_iter > 0:
            kernel = np.ones((morphology_close_kernel_size, ) * 2, np.uint8)
            merged_bin = cv2.morphologyEx(merged_bin, cv2.MORPH_CLOSE, kernel, iterations=morphology_close_kernel_iter)
        return merged_bin

    @staticmethod
    def get_sub_window(images, center_index, window_size):
        assert window_size % 2 == 1
        half_window = window_size // 2
        assert center_index >= half_window
        assert center_index < len(images) - half_window
        window_images = images[center_index - half_window:
                               center_index + half_window + 1]
        return window_images

    def process(self, images):
        """

        :param images:
        :return: List of binarized images in the window.
        """
        return self._get_merged_binary_at(images, **self.params)


class ImageAverage(ImageProcessor):
    def __init__(self):
        self.params = None
        super().__init__()

    def process(self, images):
        avg_img = np.stack(images, axis=-1).mean(axis=-1).astype('uint8')
        return avg_img


class ImageDriftCorrection(ImageProcessor):
    def __init__(self, img_shape, target_cropped_size=1200, max_shift=50, 
                 rebase_shift=10, rebase_steps=2000, i_gpu=-1):
        self.params = dict()
        self.target_cropped_size = target_cropped_size
        self.maxshift = np.array([max_shift, max_shift])
        self.rebase_shift = np.array([rebase_shift, rebase_shift])
        self.rebase_steps = rebase_steps
        self.rebase_step_counting = 0
        self.accumulated_correction = None
        self.last_img = None
        self.last_mean_on_x = None
        self.last_mean_on_y = None
        self.corrections = []
        self.img_shape = tuple(img_shape)
        self.i_gpu = i_gpu
        super().__init__()

    def rebase(self, img, cur_shift=None):
        self.last_img = img
        self.last_mean_on_x = img.mean(axis=1)
        self.last_mean_on_y = img.mean(axis=0)
        self.rebase_step_counting = 0
        if cur_shift is None:
            self.accumulated_correction = np.array([0.0, 0.0])
            self.corrections = []
        else:
            self.accumulated_correction = cur_shift

    def crop_image(self, img):
        img_shift = np.array(self.corrections[-1][0])
        top, left = ((np.array(img.shape) // 2 - self.target_cropped_size // 2) + img_shift).astype("int32")
        cropped_img = img[top:top + self.target_cropped_size, left:left + self.target_cropped_size]
        return cropped_img

    def process(self, img):
        if img.shape != self.img_shape:
            if len(self.corrections) > 0:
                self.corrections.append(copy.deepcopy(self.corrections[-1]))
            else:
                self.corrections.append([[0.0, 0.0], [0.0, 0.0]])
            return None
        if self.accumulated_correction is None:
            self.rebase(img)
        self.rebase_step_counting += 1

        if self.i_gpu >= 0:
            with cupy.cuda.Device(self.i_gpu):
                img1 = cupy.array(self.last_img)
                img2 = cupy.array(img)
                step_shift = gpu_phase_cross_correlation(img1, img2, normalization=None)[0]
            step_shift = -np.array(list(step_shift), dtype=np.int32)
        else:
            step_shift = -cpu_phase_cross_correlation(self.last_img, img, normalization=None)[0]
        img_shift = self.accumulated_correction + step_shift
        top, left = ((np.array(img.shape) // 2 - self.target_cropped_size // 2) + img_shift).astype("int32")
        cropped_img = img[top:top + self.target_cropped_size, left:left + self.target_cropped_size]
        self.corrections.append([img_shift.tolist(), step_shift.tolist()])
        if (np.abs(step_shift) >= self.rebase_shift).any() or self.rebase_step_counting >= self.rebase_steps:
            self.rebase(img, img_shift)
        return cropped_img


class ImageMaskShrinkAndShift(ImageProcessor):

    def __init__(self, param_file=None):
        if param_file is None:
            param_file = os.path.join(module_dir, "data", "mask_fix.json")
        self.params = self.load_params(param_file)
        super().__init__()


    @staticmethod
    def _shrink_shift_mask(img, shrink=0, shift=(0, 0)):
        shrink_kernel = np.ones((shrink, shrink), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, shrink_kernel, iterations=1)
        
        top1, left1 = np.max([-np.array(shift), (0, 0)], axis=0)
        bottom1, right1 = np.array(img.shape) - np.abs(shift) + np.array([top1, left1])
        top2, left2 = np.max([shift, (0, 0)], axis=0)
        bottom2, right2 = np.array(img.shape) - np.abs(shift) + np.array([top2, left2])
        
        img2 = np.zeros_like(img)
        img2[top2: bottom2, left2:right2] = img[top1: bottom1, left1: right1]
        
        return img2

    def process(self, img):
        img2 = self._shrink_shift_mask(img, **self.params)
        return img2
