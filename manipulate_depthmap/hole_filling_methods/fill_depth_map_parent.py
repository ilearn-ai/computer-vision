import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import (mean_squared_error, normalized_root_mse,
                             structural_similarity)

from manipulate_depthmap.utilities.auxiliar_methods import round_up_100
from manipulate_depthmap.utilities.plotting_utilities.radar_chart_utilities import \
    ComplexRadar


class FillDepthMap:
    def __init__(self, name: str) -> None:
        """
        Constructor method.
        """
        self.name = name
        self.metrics = {
            "mse": np.nan,  # mse = mean squared error
            "rmse": np.nan,  # rmse = (normalized) mean squared error
            "mae": np.nan,  # mae = maximum absolute error
            "ssim": np.nan,  # ssim = structural similarity
            "elapsed time": np.nan,
        }

    def fill_holes_with_timer(self, depth_map: pd.DataFrame) -> np.ndarray:
        """
        Fill the depth values holes (represented by zeros) in an sparse depth map.

        :param depth_map: input sparse depth map
        :return filled_depth_map: modified input depth map after hole completion.
        """
        start_time = time.time()
        filled_depth_map = self._fill_holes(depth_map)
        self.metrics["elapsed time"] = time.time() - start_time
        return filled_depth_map

    def _fill_holes(self, depth_map):
        raise NotImplementedError("Method not implemented")

    def validate_input_depth_map(
        self, depth_map: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Transform input depth map type and content to make it available for use in hole filling methods.

        : param depth_map: input sparse depth map to be filled up.
        : return depth_map: input depth map prepared for hole filling methods.
        """
        if isinstance(depth_map, pd.DataFrame):
            # Change depth map to numpy format to be used by cv2.inpaint method
            depth_map = depth_map.to_numpy()
        elif not isinstance(depth_map, np.ndarray):
            raise TypeError("depth_map must be a pandas DataFrame of numpy array")

        if depth_map.dtype != np.uint8:
            # Change depth map numpy format to uint8 to make it compatible with cv2.inpaint method
            depth_map = depth_map.astype(np.uint8)

        if len(depth_map.shape) != 2:
            raise ValueError("depth_map must have 2 dimensions")

        return depth_map

    def calculate_metrics(
        self, depth_map: Union[pd.DataFrame, np.ndarray], ground_truth: pd.DataFrame
    ) -> None:
        """
        Calculate metrics to evaluate the accuracy of our depth map completion.

        :param depth_map: depth map after hole completion
        :param ground_truth: real filled depth map used as ground truth for comparison
        """
        depth_map = self.validate_input_depth_map(depth_map)
        ground_truth = ground_truth.values
        self.metrics["mse"] = mean_squared_error(ground_truth, depth_map)
        self.metrics["rmse"] = normalized_root_mse(ground_truth, depth_map)
        self.metrics["mae"] = np.mean(np.abs(ground_truth - depth_map))
        self.metrics["ssim"] = structural_similarity(ground_truth, depth_map)

    def plot_metrics(self) -> None:
        """
        Plot metrics to evaluate the accuracy of our depth map completion.
        """
        kpis = self.metrics.keys()
        scores = list(self.metrics.values())

        if np.any(np.isnan(scores)):
            raise ValueError("Make sure that performance metrics have been calculated!")

        method = self.name
        ranges = [(0, round_up_100(kpi_score)) for kpi_score in scores]
        fig = plt.figure(figsize=(6, 6))
        radar = ComplexRadar(fig, kpis, ranges)
        radar.plot(scores)
        radar.fill(scores, alpha=0.2)
        plt.legend(f"{method}")
        plt.show()
