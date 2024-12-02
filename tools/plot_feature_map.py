import numpy as np
import matplotlib.pyplot as plt


def plot_feature_map(file_path, feature_map_index):

    # Load the numpy array from the specified file
    conv_features = np.load(file_path)

    # Check if the specified feature map index is within the range of available feature maps
    if feature_map_index < 0 or feature_map_index >= conv_features.shape[0]:
        print(f"Feature map index {feature_map_index} is out of range. Please specify a valid index.")
        return

    # Extract the specified feature map
    feature_map = conv_features[feature_map_index]

    # Plot the feature map
    plt.imshow(feature_map, cmap='viridis')
    # plt.title(f'Feature Map {feature_map_index + 1}')
    # plt.colorbar()
    plt.axis('off')  # This line omits the axes
    plt.show()


# Example usage
file_path = '../runs/obb/predict4/IMG_20210818_143837/stage0_Conv_features.npy'  # Update this path to your .npy file
feature_map_index = 0  # Change this to the index of the feature map you want to plot
plot_feature_map(file_path, feature_map_index)
