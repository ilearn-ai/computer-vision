import numpy as np
from scipy.interpolate import griddata


def fill_holes_in_sparse_map_using_interpolation(sparse_depth_map, method):
    filled_depth_map = sparse_depth_map.interpolate(method)
    return filled_depth_map
