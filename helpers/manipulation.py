import torch
from enum import Enum
from torchvision import transforms

# CIFAR10 Normalization values
normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
denormalize = transforms.Normalize(mean=[-0.49139968 / 0.24703233, -0.48215827 / 0.24348505, -0.44653124 / 0.26158768],
                                   std=[1 / 0.24703233, 1 / 0.24348505, 1 / 0.26158768])

class DatasetNormalizations(Enum):
    CIFAR10_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR10_STD = [0.24703233, 0.24348505, 0.26158768]


def create_random_image(image_size, mean, std):
    """
    Creates a random image from a defined mean and std normal distribution.
    Used to create more accurate random images that are built off the models
    dataset it was trained on. Mean and std must be the same length. This will
    be used to give the images its color channels. Mean of length 3 means 3 channels.

    :param image_size: Tuple of the 2D image size
    :param mean: The mean of the distribution
    :param std:  The standard deviation of the distribution
    :return: image - The created image
    """
    channels = []
    for i in range(len(mean)):  # Create each channel with the specified custom distribution
        channels.append(torch.empty((image_size[0], image_size[1])).normal_(mean=mean[i], std=std[i]))
    return torch.stack(channels)
