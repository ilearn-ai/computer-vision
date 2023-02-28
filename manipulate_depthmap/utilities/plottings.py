from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

from manipulate_depthmap.hole_filling_methods.fill_depth_map_parent import \
    FillDepthMap
from manipulate_depthmap.utilities.auxiliar_methods import round_up_100
from manipulate_depthmap.utilities.plotting_utilities.radar_chart_utilities import \
    ComplexRadar

colors = list(mcolors.TABLEAU_COLORS)


def plot_depth_map_from_img_dataframe(img_dataframe: pd.DataFrame) -> None:
    """
    Represent the depth map of a given image.
    """
    plt.matshow(img_dataframe)
    plt.colorbar()
    plt.show()


def plot_both_sparse_and_ground_truth(img_dataframe: pd.DataFrame, ground_truth_dataframe: pd.DataFrame) -> None:
    """
    Represent the depth maps of an input depth map and its corresponding ground truth.
    """
    # TODO: complete this function
    pass


def plot_single_radar_chart(labels, values, method) -> None:
    """
    Represent the radar chart of a given method.

    :param labels: KPI names to be included in the radar chart.
    :param values: values of each KPI.
    :param method: name of the method that is represented in the radar chart.
    """
    # example data
    # TODO: Make these axes adjustable to the max values of input data as in compare_methods_metrics.
    ranges = [(0, 7000), (0, 100), (0, 60)]
    # plotting
    fig1 = plt.figure(figsize=(6, 6))
    radar = ComplexRadar(fig1, labels, ranges)
    radar.plot(values)
    radar.fill(values, alpha=0.2)
    plt.legend(f"{method}")
    plt.show()


def compare_methods_metrics(filled_depth_maps: List[FillDepthMap]) -> None:
    """
    Represent the metrics of all methods in the input list in a radar chart.
    """
    values = np.vstack(
        list(filled_depth_map.metrics.values())
        for filled_depth_map in filled_depth_maps
    )
    # To define the axes ranges, pick up the biggest value of each column from values and round up it by a multiple of 100
    ranges = [(0, round_up_100(score)) for score in np.max(values, axis=0)]

    # All elements (methods) in a list share the same metrics. That is why we refer to the first one (it could be to any of them)
    labels = filled_depth_maps[0].metrics.keys()

    format_cfg = {
        "rad_ln_args": {"visible": False},
        "outer_ring": {"visible": False},
        "rgrid_tick_lbls_args": {"fontsize": 6},
        "theta_tick_lbls": {"fontsize": 9},
        "theta_tick_lbls_pad": 5,
    }

    fig = plt.figure(figsize=(6, 6), dpi=200)
    radar = ComplexRadar(fig, labels, ranges, show_scales=True, format_cfg=format_cfg)

    for idx, method in enumerate(filled_depth_maps):
        scores = list(method.metrics.values())
        radar.plot(scores, label=method.name, color=colors[idx])
        radar.use_legend()

    plt.show()
