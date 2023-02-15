import matplotlib.pyplot as plt

def plot_depth_map_from_img_dataframe(img_dataframe):
    plt.matshow(img_dataframe)
    plt.colorbar()
    plt.show()