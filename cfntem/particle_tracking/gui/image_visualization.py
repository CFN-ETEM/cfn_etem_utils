import plotly.graph_objs as go
from PIL import Image
import cv2
import numpy as np
from ict.particle_tracking.utils import numpy_array_to_plotly_image


class ImageVisualizationPlotly(object):
    def __init__(self, plot_size, margin=(25, 25, 5, 45, 2), plot_bgcolor='#c7c7c7',
                 paper_bgcolor="gray", bargap=0.05):
        self.plot_size = plot_size
        assert len(margin) == 5
        self.margin = go.layout.Margin(
            l=margin[0],
            r=margin[1],
            b=margin[2],
            t=margin[3],
            pad=margin[4])
        self.plot_bgcolor = plot_bgcolor
        self.paper_bgcolor = paper_bgcolor
        self.bargap = bargap

    def show_image(self, img, img_plot_size=None, title="Image"):
        if img_plot_size is None:
            img_plot_size = self.plot_size
        fig = go.FigureWidget()
        layout = go.Layout(
            autosize=False,
            showlegend=False,
            width=img_plot_size[0],
            height=img_plot_size[1],
            margin=self.margin,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            xaxis=dict(
                range=[0, img_plot_size[0]],
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False),
            yaxis=dict(
                range=[0, img_plot_size[1]],
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False))
        fig.layout = layout
        fig.layout.title = go.layout.Title(
            text=title,
            xref='paper',
            y=0.95)
        fig.layout.images = [numpy_array_to_plotly_image(img, img_plot_size)]
        return fig

    def show_histogram(self, img, img_plot_size=None, title="Histogram"):
        if img_plot_size is None:
            img_plot_size = self.plot_size
        x_max = np.iinfo(img.dtype).max
        fig = go.FigureWidget()
        layout = go.Layout(
            width=img_plot_size[0],
            height=img_plot_size[1],
            bargap=self.bargap,
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False),
            xaxis=dict(
                range=[0, x_max],
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=True),
            margin=self.margin)
        fig.layout = layout
        fig.layout.title = go.layout.Title(
            text=title,
            xref='paper',
            y=0.95)
        density, hist_edges = np.histogram(img.ravel(), bins=15, range=[0, x_max], density=True)
        fig.add_histogram(histfunc="sum",
                          x=hist_edges[:-1] + (x_max // 15) / 2,
                          y=density,
                          xbins=dict(start=0, end=x_max, size=x_max // 15))
        return fig

    def select_particles(self, img_raw, img_bin, img_plot_size=None, title="Select Particles"):
        fig = self.show_image(img_raw, img_plot_size, title)
        contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_centers = np.array([cnt[:, 0, ::-1].mean(axis=0) for cnt in contours])

        def selection_fn(trace, points, selector):
            center = [np.array(selector.xrange).mean(), np.array(selector.yrange).mean()]
            c_on_img = (fig.layout.yaxis.range[1] - center[1], center[0])
            cnt_id = np.linalg.norm(contour_centers - np.array(c_on_img), axis=-1).argmin()
            x, y = list(contours[cnt_id][:, 0, :].T)
            scatter.x, scatter.y = x, fig.layout.yaxis.range[1] - y
            fig.layout.annotations = [
                dict(
                    x=center[0],
                    y=center[1],
                    xref='x',
                    yref='y',
                    text=str(cnt_id),
                    showarrow=True,
                    arrowhead=7,
                    font=dict(
                        family='Courier New, monospace',
                        size=16,
                        color='orange'),
                    ax=0,
                    ay=-5)]

        fig.layout.clickmode = 'event'
        fig.layout.dragmode = 'select'
        fig.layout.xaxis.range = [0, img_raw.shape[1]]
        fig.layout.yaxis.range = [0, img_raw.shape[0]]
        fig.add_scatter(x=[50, 55], mode='lines+markers', marker=dict(size=1))
        scatter = fig.data[0]
        scatter.on_selection(selection_fn)
        return fig


    @staticmethod
    def infer_plot_size_from_image(img, target_width=400):
        scale = 1.0 / round(img.shape[1] / target_width)
        return tuple(reversed((np.array(img.shape) * scale).astype(np.uint32)))
