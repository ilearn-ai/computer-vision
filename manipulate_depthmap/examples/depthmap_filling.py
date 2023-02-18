from pathlib import Path

import pandas as pd

from manipulate_depthmap.hole_filling_methods.interpolation_methods import (
    fill_holes_in_sparse_map_using_interpolation,
)
from manipulate_depthmap.utilities.plot_utilities import (
    plot_depth_map_from_img_dataframe,
)

sparse_map_url = (
    Path(__file__).parent.parent / "source_images" / "Rosemary" / "sparse_depth.dat"
)
sparse_map = pd.read_csv(sparse_map_url, header=None)
plot_depth_map_from_img_dataframe(sparse_map)

method = "linear"
filled_depth_map = fill_holes_in_sparse_map_using_interpolation(sparse_map, method)
plot_depth_map_from_img_dataframe(filled_depth_map)
