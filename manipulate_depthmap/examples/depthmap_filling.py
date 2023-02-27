from pathlib import Path

import cv2
import pandas as pd

from manipulate_depthmap.hole_filling_methods.inpaint_methods.diffusion_based import \
    DiffusionInpaintMethod
from manipulate_depthmap.hole_filling_methods.inpaint_methods.fast_marching_based import \
    FastMarchingInpaintMethod
from manipulate_depthmap.hole_filling_methods.inpaint_methods.patch_based import \
    PatchInpaintMethod
from manipulate_depthmap.hole_filling_methods.interpolation_methods import \
    InterpolationMethod
from manipulate_depthmap.utilities.plottings import (
    compare_methods_metrics, plot_depth_map_from_img_dataframe)

image_name = "Rosemary"

sparse_map_url = (
    Path(__file__).parent.parent
    / "source_images"
    / f"{image_name}"
    / "sparse_depth.dat"
)
sparse_map = pd.read_csv(sparse_map_url, header=None)
plot_depth_map_from_img_dataframe(sparse_map)

ground_truth_url = (
    Path(__file__).parent.parent / "source_images" / f"{image_name}" / "gt_depth.dat"
)
ground_truth = pd.read_csv(ground_truth_url, header=None)
plot_depth_map_from_img_dataframe(ground_truth)

source_img_url = (
    Path(__file__).parent.parent / "source_images" / f"{image_name}" / "image.png"
)
source_img = cv2.imread(str(source_img_url))

# Filling though interpolation
method = "nearest"
interpolation_method = InterpolationMethod(interpolation_type=method)
filled_depth_map_interpol = interpolation_method.fill_holes_with_timer(
    depth_map=sparse_map
)
interpolation_method.calculate_metrics(
    depth_map=filled_depth_map_interpol, ground_truth=ground_truth
)
plot_depth_map_from_img_dataframe(filled_depth_map_interpol)

# Filling through patch inpainting
inpaint_radius = 2
patch_inpaint_method = PatchInpaintMethod(inpaint_radius=inpaint_radius)
filled_depth_map_patch_inpaint = patch_inpaint_method.fill_holes_with_timer(
    depth_map=sparse_map
)
patch_inpaint_method.calculate_metrics(
    depth_map=filled_depth_map_patch_inpaint, ground_truth=ground_truth
)
plot_depth_map_from_img_dataframe(filled_depth_map_patch_inpaint)

# Filling through diffusion inpainting
diffusion_inpaint_method = DiffusionInpaintMethod()
filled_depth_map_diffusion_inpaint = diffusion_inpaint_method.fill_holes_with_timer(
    depth_map=sparse_map
)
diffusion_inpaint_method.calculate_metrics(
    depth_map=filled_depth_map_diffusion_inpaint, ground_truth=ground_truth
)
plot_depth_map_from_img_dataframe(filled_depth_map_diffusion_inpaint)

# Filling through fast-marching inpainting
inpaint_radius = 5
fast_marching_inpaint_method = FastMarchingInpaintMethod(inpaint_radius=inpaint_radius)
filled_depth_map_fmm_inpaint = fast_marching_inpaint_method.fill_holes_with_timer(
    depth_map=sparse_map
)
fast_marching_inpaint_method.calculate_metrics(
    depth_map=filled_depth_map_fmm_inpaint, ground_truth=ground_truth
)
plot_depth_map_from_img_dataframe(filled_depth_map_fmm_inpaint)

methods_studied = [
    interpolation_method,
    patch_inpaint_method,
    diffusion_inpaint_method,
    fast_marching_inpaint_method,
]
compare_methods_metrics(methods_studied)
