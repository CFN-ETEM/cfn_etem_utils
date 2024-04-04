import cv2
from PIL import Image
import plotly.graph_objs as go
import numpy as np


def numpy_array_to_plotly_image(img, img_plot_size=None):
    img_params = dict(
        opacity=1.0,
        layer="below",
        sizing="stretch",
        x=0.0,
        y=1.0,
        sizex=1.0,
        sizey=1.0
    )
    if img_plot_size is not None:
        img = cv2.resize(img, img_plot_size, cv2.INTER_AREA)
    img = Image.fromarray(img)
    return go.layout.Image(source=img, **img_params)

def color_name_to_rgba(name):
    name2rgba = {"blue":[0, 0, 255, 255],
                 "green": [0, 255, 0, 255],
                 "red": [255, 0, 0, 255]}
    if name in name2rgba:
        color = name2rgba[name]
    elif "#" in name:
        r = int(name[1:3], base=16)
        g = int(name[3: 5], base=16)
        b = int(name[5: 7], base= 16)
        color = [r, g, b, 255]
    else:
        raise ValueError("Can't understand color name \"{}\"".format(name))
    return color

def colored_detection_boundary(img, img_plot_size, color, thickness=1):
    assert isinstance(color, list) and len(color) == 4
    if (isinstance(img_plot_size, list) or isinstance(img_plot_size, tuple)) and img_plot_size[0] > 0:
        img_resized = cv2.resize(img, tuple(reversed(img_plot_size)), cv2.INTER_AREA)
    else:
        img_resized = img
    contours, _ = cv2.findContours(img_resized, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img_plot_size, np.uint8)
    cv2.drawContours(mask, contours, contourIdx=-1, color=255,thickness=thickness)
    mask_color = np.zeros(list(img_plot_size) + [4], np.uint8)
    mask_color[mask == 255] = color
    return mask_color


def estimate_detection_error(merged_image, images_window):
    fake_upper_area = np.stack(images_window, axis=-1).max(axis=-1).astype('uint8')
    fake_lower_area = np.stack(images_window, axis=-1).min(axis=-1).astype('uint8')
    upper_area, pred_area, lower_area = fake_upper_area.sum(), merged_image.sum(), fake_lower_area.sum()
    area_trust_ratio = (
    round((upper_area - pred_area) / upper_area, 3), -round((pred_area - lower_area) / lower_area, 3))

    upper_contours, _ = cv2.findContours(fake_upper_area, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(merged_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lower_contours, _ = cv2.findContours(fake_lower_area, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    upper_number, pred_number, lower_number = len(upper_contours), len(pred_contours), len(lower_contours)
    number_trust_ratio = (max(round((upper_number - pred_number) / upper_number, 3), 0.0),
                          min(round(-((pred_number - lower_number) / lower_number), 3), 0.0))
    trust_ratio = dict(area=area_trust_ratio, number=number_trust_ratio)
    return trust_ratio