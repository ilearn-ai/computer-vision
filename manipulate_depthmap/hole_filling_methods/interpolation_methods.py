import numpy as np
import pandas as pd

from manipulate_depthmap.hole_filling_methods.fill_depth_map_parent import \
    FillDepthMap


class InterpolationMethod(FillDepthMap):
    def __init__(self, interpolation_type: str = 'linear'):
        self.interpolation_type = interpolation_type
        super().__init__("interpolation")

    def fill_holes(self, depth_map: pd.DataFrame) -> pd.DataFrame:
        depth_map.replace(0, np.nan, inplace=True)
        filled_depth_map = depth_map.interpolate(self.interpolation_type)
        return filled_depth_map