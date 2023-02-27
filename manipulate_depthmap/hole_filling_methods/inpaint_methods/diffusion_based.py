import numpy as np
import pandas as pd
from skimage.restoration import inpaint

from manipulate_depthmap.hole_filling_methods.fill_depth_map_parent import FillDepthMap


class DiffusionInpaintMethod(FillDepthMap):
    def __init__(self):
        super().__init__("diffuse_inpaint")

    def _fill_holes(self, depth_map: pd.DataFrame) -> np.ndarray:
        depth_map = self.validate_input_depth_map(depth_map)
        inpaint_mask = np.uint8(depth_map == 0)

        depth_map = np.ascontiguousarray(depth_map)
        inpaint_mask = np.ascontiguousarray(inpaint_mask)
        filled_depth_map = inpaint.inpaint_biharmonic(depth_map, inpaint_mask)
        return 255 * filled_depth_map
