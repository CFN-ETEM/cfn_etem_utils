import abc
import re, json
from collections import OrderedDict
from inspect import signature

from ipywidgets import interactive_output, VBox, HBox, IntSlider, FloatSlider, \
    Checkbox, IntRangeSlider, ColorPicker, Button, IntProgress, Tab
from ipywidgets import Layout as IpyLayout
from IPython.display import display

import numpy as np

from cfntem.particle_tracking.image_processing import ImageEnhance, ImageBinarization, ImageMerge
from cfntem.particle_tracking.gui.image_visualization import ImageVisualizationPlotly
from cfntem.particle_tracking.io import load_dm4_file
from cfntem.particle_tracking.utils import numpy_array_to_plotly_image, color_name_to_rgba, \
    colored_detection_boundary


class ParamTuner(abc.ABC):
    def __init__(self):
        self._progress_bar = IntProgress(value=1, min=0, max=10, description="{:<55s}".format("waiting for input"),
            layout=IpyLayout(width='98%'), style={'description_width': 'initial'})

    @abc.abstractmethod
    def calculate(self, **kwargs):
        pass

    @abc.abstractmethod
    def erase_plot_box(self):
        pass

    @abc.abstractmethod
    def render_plot_box(self):
        pass

    def tune(self, **kwargs):
        self.progress_bar.value = 1
        self.progress_bar.description = "{:<55s}".format("Erase Previous Result")
        self.erase_plot_box()

        self.progress_bar.value = 2
        self.progress_bar.description = "{:<55s}".format("Calculating")
        self.calculate(**kwargs)

        self.progress_bar.value = 8
        self.progress_bar.description = "{:<55s}".format("Redraw New Result")
        self.render_plot_box()

        self.progress_bar.value = 10
        self.progress_bar.description = "{:<55s}".format("Done")


    def show(self, widget_width="5in", full_width="10in", continuous_update=False):
        param_dict = self.target_param_and_widgets
        self.interactive_out = interactive_output(self.tune, param_dict)
        if not continuous_update:
            for c in param_dict.values():
                c.continuous_update = False
        full_box = HBox(
            [self.plot_box,
             VBox([self.progress_bar, self.widget_box, self.interactive_out],
                  layout=IpyLayout(width=widget_width,
                                   justify_content='space-between',
                                   align_items="center"))],
            layout=IpyLayout(width=full_width,
                             justify_content='space-between',
                             align_items="center"))
        return full_box

    def save_params(self, filename=None):
        if filename is None:
            re.sub(r'.*\.', '', str(self.__class__)).replace("'>", "").replace("Tuner", "") + ".json"
        with open(filename, "w") as f:
            json.dump(self.parameters, f, indent=4, sort_keys=True)

    @property
    def parameters(self):
        param_dict = self.target_param_and_widgets
        kwargs = {k: v.value for k, v in param_dict.items()}
        return kwargs


    @property
    @abc.abstractmethod
    def plot_box(self):
        pass

    @property
    @abc.abstractmethod
    def widget_box(self):
        pass


    @property
    def progress_bar(self):
        return self._progress_bar

    @property
    @abc.abstractmethod
    def target_param_and_widgets(self):
        pass


class WidgetBox(abc.ABC):
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def params_dict(self):
        pass

    @property
    @abc.abstractmethod
    def box(self):
        pass


class ImageEnhanceWidgetBox(WidgetBox):

    def __init__(self, reverse_color, pre_bilat, bilat_r_space, bilat_r_color,
            use_hist_eqz, hist_clip_limit, hist_grid_size, post_bilat,
            description_width='40%'):
        content_layout = IpyLayout(width='98%')
        content_style = {'description_width': description_width}
        self._params_dict = OrderedDict()
        self._params_dict["reverse_color"] = Checkbox(style=content_style,
            value=reverse_color, indent=True, layout=content_layout,
            description="Reverse Color")
        self._params_dict["pre_bilat"] = Checkbox(style=content_style,
            value=pre_bilat, indent=True, layout=content_layout,
            description="Apply Bilateral Filter Before Smoothing")
        self._params_dict["post_bilat"] = Checkbox(style=content_style,
            value=post_bilat, indent=True, layout=content_layout,
            description="Apply Bilateral Filter After Smoothing")
        self._params_dict["bilat_r_space"] = IntSlider(style=content_style,
            value=bilat_r_space, min=1, step=1, max=30, layout=content_layout,
            description="Bilateral Filter Spatial Radius")
        self._params_dict["bilat_r_color"] = IntSlider(style=content_style,
            value=bilat_r_color, min=1, step=1, max=255, layout=content_layout,
            description="Bilateral Filter Color Range")
        self._params_dict["use_hist_eqz"] = Checkbox(style=content_style,
            value=use_hist_eqz, indent=True, layout=content_layout,
            description="Apply Histogram Equalization")
        self._params_dict["hist_clip_limit"] = FloatSlider(style=content_style,
            value=hist_clip_limit, min=0.2, step=0.2, max=20.0, layout=content_layout,
            description="Contrast Limiting Threshold")
        self._params_dict["hist_grid_size"] = IntSlider(style=content_style,
            value=hist_grid_size, min=1, step=1, max=25, layout=content_layout,
            description="Histogram Equalization Tile Size")
        sig = signature(ImageEnhanceWidgetBox.__init__)
        sig_param_set = set(sig.parameters.keys())
        sig_param_set.remove("self")
        sig_param_set.remove("description_width")
        assert set(self._params_dict) == sig_param_set
        self._box = VBox(list(self._params_dict.values()), layout=content_layout)
        super().__init__()

    @property
    def box(self):
        return self._box

    @property
    def params_dict(self):
        return dict(self._params_dict)


class ImageEnhanceTuner(ParamTuner):

    def __init__(self, img_orig, img_plot_size, processor=ImageEnhance()):
        self.img_orig = img_orig.copy()
        self.img_plot_size = img_plot_size
        self.processor = processor
        self.img_processed = self.processor.process(self.img_orig)
        self.vis = ImageVisualizationPlotly(self.img_plot_size)
        self.orig_img_fig = self.vis.show_image(self.img_orig, self.img_plot_size, title="Original Image")
        self.orig_hist_fig = self.vis.show_histogram(self.img_orig, self.img_plot_size, title="Original Histogram")
        self.processed_img_fig = self.vis.show_image(self.img_processed, self.img_plot_size, title="Processed Image")
        self.processed_hist_fig = self.vis.show_histogram(self.img_processed, self.img_plot_size, title="Processed Histogram")
        self._plot_box = VBox([HBox([self.orig_img_fig, self.orig_hist_fig],
                                    layout=IpyLayout(width="{}px".format(self.img_plot_size[0] * 2))),
                               HBox([self.processed_img_fig, self.processed_hist_fig],
                                    layout=IpyLayout(width="{}px".format(self.img_plot_size[0] * 2)))],
                              layout=IpyLayout(height="{}px".format(self.img_plot_size[1] * 2 + 5),
                                               width="{}px".format(self.img_plot_size[0] * 2 + 40),
                                               justify_content='space-between'))
        wb = ImageEnhanceWidgetBox(**self.processor.params)
        self._widget_box = wb.box
        self._target_param_and_widgets = wb.params_dict
        self.dummy_image_white = numpy_array_to_plotly_image(
            np.full(list(reversed(self.img_plot_size)), fill_value=255,
                    dtype=np.uint8))
        self.dummy_hist = np.full((15,), fill_value=0.0, dtype=np.float64)
        super().__init__()

    def calculate(self, **kwargs):
        assert set(kwargs.keys()) == set(self.processor.params.keys())
        self.processor.params.update(kwargs)
        self.img_processed = self.processor.process(self.img_orig)


    def erase_plot_box(self):
        self.processed_img_fig.layout.images = [self.dummy_image_white]
        self.processed_hist_fig.data[0].y = self.dummy_hist

    def render_plot_box(self):
        self.processed_img_fig.layout.images = [
            numpy_array_to_plotly_image(self.img_processed, self.img_plot_size)]

        hist_density, _ = np.histogram(self.img_processed.ravel(), bins=15,
                                       range=[0, 255], density=True)
        self.processed_hist_fig.data[0].y = hist_density

    @property
    def plot_box(self):
        return self._plot_box

    @property
    def widget_box(self):
        return self._widget_box

    @property
    def target_param_and_widgets(self):
        return self._target_param_and_widgets


class ImageBinarizationTabPanelWidgetBox(WidgetBox):

    def __init__(self, edge_thr,use_l2_gradient, edge_fix_kernel_size, sobel_size,
            edge_fix_iter, denoise_kernel_size, denoise_iter, exclude_kernel_size,
            exclude_iter, expansion_kernel_size, expansion_iter, color,
            description_width='40%'):

        content_layout = IpyLayout(width='98%')
        content_style = {'description_width': description_width}
        self._params_dict = OrderedDict()

        self._params_dict["edge_thr"] = IntRangeSlider(
            value=edge_thr, min=0, step=1, max=255, layout=content_layout,
            description="Canny Edge Threshold", style=content_style)

        self._params_dict["use_l2_gradient"] = Checkbox(
            value=use_l2_gradient, indent=True, layout=content_layout,
            description="Use L2 Gradient", style=content_style)

        self._params_dict["sobel_size"] = IntSlider(
            value=sobel_size, min=3, step=2, max=11, layout=content_layout,
            description="Canny Sobel Kernel Size", style=content_style)

        self._params_dict["edge_fix_kernel_size"] = IntSlider(
            value=edge_fix_kernel_size, min=1, step=1, max=10, layout=content_layout,
            description="Edge Fix Kernel Size", style=content_style)

        self._params_dict["edge_fix_iter"] = IntSlider(
            value=edge_fix_iter, min=1, step=1, max=10, layout=content_layout,
            description="Edge Fix Iterations", style=content_style)

        self._params_dict["denoise_kernel_size"] = IntSlider(
            value=denoise_kernel_size, min=1, step=1, max=15, layout=content_layout,
            description="Denoise Kernel Size", style=content_style)

        self._params_dict["denoise_iter"] = IntSlider(
            value=denoise_iter, min=1, step=1, max=10, layout=content_layout,
            description="Denoise Iterations", style=content_style)

        self._params_dict["exclude_kernel_size"] = IntSlider(
            value=exclude_kernel_size, min=1, step=1, max=40, layout=content_layout,
            description="Edge Exclusion Kernel Size", style=content_style)

        self._params_dict["exclude_iter"] = IntSlider(
            value=exclude_iter, min=1, step=1, max=5, layout=content_layout,
            description="Edge Exclusion Iterations", style=content_style)

        self._params_dict["expansion_kernel_size"] = IntSlider(
            value=expansion_kernel_size, min=1, step=1, max=20, layout=content_layout,
            description="Expansion Kernel Size", style=content_style)

        self._params_dict["expansion_iter"] = IntSlider(
            value=expansion_iter, min=0, step=1, max=5, layout=content_layout,
            description="Expansion Iterations", style=content_style)

        self.color = ColorPicker(value=color,
            description="Marker Color", style=content_style)

        self._box = VBox(list(self._params_dict.values()) + [self.color], layout=content_layout)

        sig = signature(ImageBinarizationTabPanelWidgetBox.__init__)
        sig_param_set = set(sig.parameters.keys())
        sig_param_set.remove("self")
        sig_param_set.remove(("color"))
        sig_param_set.remove("description_width")
        assert set(self._params_dict) == sig_param_set

        self.content_layout = content_layout
        self.content_style = content_style

        super().__init__()

    def hide_exclude_control(self):
        self._params_dict["exclude_kernel_size"].layout = IpyLayout(visibility="hidden")
        self._params_dict["exclude_iter"].layout = IpyLayout(visibility="hidden")

    def append_copy_button(self, from_step_number, from_widgets):
        def on_button_copy_clicked(button):
            for b1_w, b2_w in zip(self.params_dict.values(),
                                  from_widgets.params_dict.values()):
                b1_w.value = b2_w.value

        self.copy_button = Button(description="Copy Attempt {}".format(from_step_number),
                                  layout=self.content_layout)
        self.box.children = self.box.children + (self.copy_button,)
        self.copy_button.on_click(on_button_copy_clicked)

    @property
    def params_dict(self):
        return dict(self._params_dict)

    @property
    def box(self):
        return self._box


class ImageBinarizationTabWidgetBox(WidgetBox):

    def __init__(self, description_width='40%', **kwargs):
        sub_sig = signature(ImageBinarizationTabPanelWidgetBox.__init__)
        sub_key_list = list(sub_sig.parameters.keys())
        sub_key_list.remove("self")
        sub_key_list.remove("color")
        sub_key_list.remove('description_width')
        full_key_list = ['a{}_{}'.format(attempt, k) for attempt in range(1, 4) for k in sub_key_list]
        assert set(kwargs.keys()) == set(full_key_list)
        self._params_dict = OrderedDict()
        content_layout = IpyLayout(width='98%')

        self.panels = []

        for attempt, color in zip(range(1, 4), ["green", "blue", "red"]):
            sub_kwargs_keys = ['a{}_{}'.format(attempt, k) for k in sub_key_list]
            sub_kwargs = {fk: kwargs[wk] for fk, wk in zip(sub_key_list, sub_kwargs_keys)}
            panel = ImageBinarizationTabPanelWidgetBox(**sub_kwargs, color=color, description_width=description_width)
            if attempt == 1:
                panel.hide_exclude_control()
            if attempt > 1:
                panel.append_copy_button(attempt - 1,  self.panels[attempt-2])
            for k, v in panel.params_dict.items():
                self._params_dict['a{}_{}'.format(attempt, k)] = v
            self.panels.append(panel)

        widget_tab = Tab(layout=content_layout)
        widget_tab.children = [panel.box for panel in self.panels]
        for i in range(len(widget_tab.children)):
            widget_tab.set_title(i, "Attempt {:d}".format(i + 1))

        self._box = widget_tab

        super().__init__()

    @property
    def colors(self):
        return [panel.color.value for panel in self.panels]

    @property
    def params_dict(self):
        return dict(self._params_dict)

    @property
    def box(self):
        return self._box


class ImageBinarizationTuner(ParamTuner):


    def __init__(self, img_orig, img_plot_size, processor=ImageBinarization()):
        self.img_orig = img_orig.copy()
        self.img_plot_size = img_plot_size
        self.processor = processor
        self.boundary_images = None
        self.vis = ImageVisualizationPlotly(self.img_plot_size)
        self.fig = self.vis.show_image(self.img_orig, self.img_plot_size,
                                       title="Detection Result")
        self._plot_box = HBox([self.fig],
            layout=IpyLayout(height="{}px".format(self.img_plot_size[1] + 5),
                             width="{}px".format(self.img_plot_size[0] + 25),
                             justify_content='space-between'))
        wb = ImageBinarizationTabWidgetBox(**self.processor.params)
        self._widget_box = wb
        self._target_param_and_widgets = wb.params_dict
        super().__init__()

    def calculate(self, **kwargs):
        assert set(kwargs.keys()) == set(self.processor.params.keys())
        self.processor.params.update(kwargs)
        _, detection_hierachy = self.processor.get_process_hiechary(self.img_orig)
        self.boundary_images = self._hierachy_to_images(detection_hierachy)

    def erase_plot_box(self):
        self.fig.layout.images = [self.fig.layout.images[0]]

    def render_plot_box(self):
        self.fig.layout.images = [self.fig.layout.images[0]] + self.boundary_images

    def _hierachy_to_images(self, detection_hierachy):
        boundary_images = []
        for i, pt in enumerate(detection_hierachy):
            color_name = self._widget_box.colors[i]
            try:
                color = color_name_to_rgba(color_name)
            except:
                self.progress_bar.description = "Error Decode Color"
                raise
            boundary = colored_detection_boundary(pt, self.img_plot_size, color)
            boundary_images.append(boundary)
        overlayed_boundary_img = np.stack(boundary_images, axis=-1).sum(axis=-1).astype('uint8')
        boundaries = numpy_array_to_plotly_image(overlayed_boundary_img)
        return [boundaries]

    @property
    def plot_box(self):
        return self._plot_box

    @property
    def widget_box(self):
        return self._widget_box.box

    @property
    def target_param_and_widgets(self):
        return self._target_param_and_widgets


class ImageMergeWidgetBox(WidgetBox):

    def __init__(self, window_size, trust_detection_rate, max_windows_size, description_width='40%'):
        content_layout = IpyLayout(width='98%')
        content_style = {'description_width': description_width}
        self._params_dict = OrderedDict()
        self._params_dict["window_size"] = IntSlider(style=content_style,
            value=window_size, min=3, step=2, max=max_windows_size, layout=content_layout,
            description="Windows Size")
        self._params_dict["trust_detection_rate"] = FloatSlider(style=content_style,
            value=trust_detection_rate, min=0.05, step=0.05, max=1.0, layout=content_layout,
            description="Detection Trust Rate")
        sig = signature(ImageMergeWidgetBox.__init__)
        sig_param_set = set(sig.parameters.keys())
        sig_param_set.remove("self")
        sig_param_set.remove("description_width")
        sig_param_set.remove("max_windows_size")
        assert set(self._params_dict) == sig_param_set
        self._box = VBox(list(self._params_dict.values()), layout=content_layout)
        super().__init__()

    @property
    def box(self):
        return self._box

    @property
    def params_dict(self):
        return dict(self._params_dict)


class ImageMergeTuner(ParamTuner):

    def __init__(self, img_file_list, img_plot_size, pre_processors,
                 processor=ImageMerge(), marker_color="blue"):
        center_index = len(img_file_list) // 2
        file_window = ImageMerge.get_sub_window(img_file_list, center_index,
                                                window_size=len(img_file_list))
        self.images = [load_dm4_file(fn, processors=pre_processors)
                       for fn in file_window]
        self.img_plot_size = img_plot_size
        self.marker_color = color_name_to_rgba(marker_color)
        self.processor = processor
        img_orig = load_dm4_file(img_file_list[center_index])
        self.vis = ImageVisualizationPlotly(self.img_plot_size)
        self.fig = self.vis.show_image(img_orig, self.img_plot_size,
                                       title="Merge Result")
        self._plot_box = HBox([self.fig],
            layout=IpyLayout(height="{}px".format(self.img_plot_size[1] + 5),
                             width="{}px".format(self.img_plot_size[0] + 25),
                             justify_content='space-between'))
        wb = ImageMergeWidgetBox(**self.processor.params, max_windows_size=len(img_file_list))
        self._widget_box = wb.box
        self._target_param_and_widgets = wb.params_dict
        super().__init__()

    def calculate(self, **kwargs):
        assert set(kwargs.keys()) == set(self.processor.params.keys())
        self.processor.params.update(kwargs)
        ws = self.processor.params["window_size"]
        img_merged = self.processor.process(
            ImageMerge.get_sub_window(self.images, len(self.images) //2, ws))
        boundary_a = colored_detection_boundary(img_merged, self.img_plot_size, self.marker_color)
        self.boundary = numpy_array_to_plotly_image(boundary_a)

    def erase_plot_box(self):
        self.fig.layout.images = [self.fig.layout.images[0]]

    def render_plot_box(self):
        self.fig.layout.images = [self.fig.layout.images[0], self.boundary]

    @property
    def plot_box(self):
        return self._plot_box

    @property
    def widget_box(self):
        return self._widget_box

    @property
    def target_param_and_widgets(self):
        return self._target_param_and_widgets