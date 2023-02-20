from typing import Union

import numpy as np
import pandas as pd

from manipulate_depthmap.utilities.plottings import plot_radar_chart


class FillDepthMap:
    def __init__(self, name):
        self.name = name
        self.mse = np.nan
        self.rmse = np.nan
        self.mae = np.nan

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
        self.mse = np.mean(np.square(ground_truth - depth_map))
        self.rmse = np.sqrt(self.mse)
        self.mae = np.mean(np.abs(ground_truth - depth_map))

    def plot_metrics(self) -> None:
        kpis = ["MSE (Mean square error)", "RMSE (Root mean square error)", "MAE (Mean absolute error)"]
        scores = [self.mse, self.rmse, self.mae]
        method = self.name
        plot_radar_chart(kpis, scores, method)