import numpy as np
import pandas as pd

from manipulate_depthmap.hole_filling_methods.fill_depth_map_parent import \
    FillDepthMap


class InterpolationMethod(FillDepthMap):
    def __init__(self, interpolation_type: str = "linear"):
        """
        Constructor method.
        """
        self.interpolation_type = interpolation_type
        super().__init__("interpolation")

    def _fill_holes(self, depth_map: pd.DataFrame) -> pd.DataFrame:
        """
        Fill the depth values holes (represented by zeros) in an sparse depth map.

        :param depth_map: input sparse depth map
        :return filled_depth_map: modified input depth map after hole completion.
        """
        depth_map.replace(0, np.nan, inplace=True)
        filled_depth_map = depth_map.interpolate(self.interpolation_type)
        return filled_depth_map
