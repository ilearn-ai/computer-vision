from pathlib import Path

import cv2
import pandas as pd

from manipulate_depthmap.hole_filling_methods.inpainting_methods import \
    InpaintingMethod
from manipulate_depthmap.hole_filling_methods.interpolation_methods import \
    InterpolationMethod
from manipulate_depthmap.utilities.plottings import \
    plot_depth_map_from_img_dataframe

image_name = "Rosemary"

sparse_map_url = (
    Path(__file__).parent.parent / "source_images" / f"{image_name}" / "sparse_depth.dat"
)
sparse_map = pd.read_csv(sparse_map_url, header=None)
plot_depth_map_from_img_dataframe(sparse_map)

ground_truth_url = (Path(__file__).parent.parent / "source_images" / f"{image_name}" / "gt_depth.dat")
ground_truth = pd.read_csv(ground_truth_url, header=None)
plot_depth_map_from_img_dataframe(ground_truth)

source_img_url = (Path(__file__).parent.parent / "source_images" / f"{image_name}" / "image.png")
source_img = cv2.imread(str(source_img_url))

# Filling though interpolation
method = "nearest"
interpolation_method = InterpolationMethod(interpolation_type=method)
filled_depth_map_interpol = interpolation_method.fill_holes(depth_map=sparse_map)
interpolation_method.calculate_metrics(depth_map=filled_depth_map_interpol, ground_truth=ground_truth)
plot_depth_map_from_img_dataframe(filled_depth_map_interpol)

# Filling through inpainting
inpaint_radius = 2
inpainting_method = InpaintingMethod(inpaint_radius=inpaint_radius)
filled_depth_map_inpaint = inpainting_method.fill_holes(depth_map=sparse_map)
inpainting_method.calculate_metrics(depth_map=filled_depth_map_inpaint, ground_truth=ground_truth)
plot_depth_map_from_img_dataframe(filled_depth_map_inpaint)

interpolation_method.plot_metrics()
inpainting_method.plot_metrics()