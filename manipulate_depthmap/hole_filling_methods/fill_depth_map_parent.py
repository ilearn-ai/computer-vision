from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from manipulate_depthmap.utilities.auxiliar_methods import round_up_100
from manipulate_depthmap.utilities.plotting_utilities.radar_chart_utilities import ComplexRadar


class FillDepthMap:
    def __init__(self, name):
        self.name = name
        self.metrics = {
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
        }

    def fill_holes(self, depth_map):
        raise NotImplementedError("Method not implemented")

    def validate_input_depth_map(self, depth_map):
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

    def calculate_metrics(self, depth_map: Union[pd.DataFrame, np.ndarray], ground_truth: pd.DataFrame) -> None:
        depth_map = self.validate_input_depth_map(depth_map)
        ground_truth = ground_truth.values
        self.metrics['mse'] = np.mean(np.square(ground_truth - depth_map))
        self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
        self.metrics['mae'] = np.mean(np.abs(ground_truth - depth_map))

    def plot_metrics(self) -> None:
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
        