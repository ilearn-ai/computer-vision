import matplotlib.pyplot as plt
from manipulate_depthmap.utilities.plotting_utilities.radar_chart_utilities import ComplexRadar


def plot_depth_map_from_img_dataframe(img_dataframe):
    plt.matshow(img_dataframe)
    plt.colorbar()
    plt.show()


def plot_both_sparse_and_ground_truth(img_dataframe, ground_truth_dataframe):
    pass


def plot_radar_chart(labels, values, method):
    # example data
    ranges = [(0, 7000), (0, 100), (0, 60)]            
    # plotting
    fig1 = plt.figure(figsize=(6, 6))
    radar = ComplexRadar(fig1, labels, ranges)
    radar.plot(values)
    radar.fill(values, alpha=0.2)
    plt.title(f"{method}")
    plt.show()