import cv2
import numpy as np
import pandas as pd

from manipulate_depthmap.hole_filling_methods.fill_depth_map_parent import \
    FillDepthMap


class PatchInpaintMethod(FillDepthMap):
    def __init__(self, inpaint_radius: str) -> None:
        """
        Constructor method.
        """
        self.inpaint_radius = inpaint_radius
        super().__init__("patch_inpaint")

    def _fill_holes(self, depth_map: pd.DataFrame) -> np.ndarray:
        """
        Fill the depth values holes (represented by zeros) in an sparse depth map.

        :param depth_map: input sparse depth map
        :return filled_depth_map: modified input depth map after hole completion.
        """
        depth_map = self.validate_input_depth_map(depth_map)
        inpaint_mask = np.uint8(depth_map == 0)
        filled_depth_map = cv2.inpaint(
            src=depth_map,
            inpaintMask=inpaint_mask,
            inpaintRadius=self.inpaint_radius,
            flags=cv2.INPAINT_TELEA,
        )
        return filled_depth_map
