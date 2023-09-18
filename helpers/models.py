# Holds methods and enums used for model interaction and logic
# Author: Bradley Gathers

import os
import torch
import helpers.manipulation as h_manipulation
from enum import Enum
from torchvision import models
from torch import nn, load, hub


class ModelTypes(Enum):
    """
    The valid model types that can be used. Simplifies selection of models.
    
    Valid values: RESNET, ALEXNET, DENSENET, EFFICIENTNET, GOOGLENET, 
    MOBILENET, SQUEEZENET.
    """
    RESNET       = 0
    ALEXNET      = 1
    DENSENET     = 2
    EFFICIENTNET = 3
    GOOGLENET    = 4
    MOBILENET    = 5
    SQUEEZENET   = 6


_hook_activations = None


# Hook function
def _get_activations():
    """
    Used when registering forward hooks to get layer activations
    :return:
    """
    def hook(model, input, output):
        global _hook_activations
        _hook_activations = output

    return hook


def _get_activation_shape():
    """
    Gets the activation
    :return: A Tuple of the size of the activation
    """
    return _hook_activations.squeeze().size()


def setup_model(model):
    """
    Takes in a model type and creates a standard and robust model. Currently 
    only setups up for the CIFAR10 dataset. Loads the respective standard and 
    robust checkpoints saved in the models directory. Raises a ValueError if 
    model type is not valid.

    :param model: Expects a enum of type ModelTypes to specified model to setup
    :return: model
    """
    curr_dir = os.path.dirname(__file__) + "/../"
    match model:
        case ModelTypes.RESNET:
            base = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1)
            base.fc = nn.Linear(base.fc.in_features, 10)

            base.load_state_dict(
                load(curr_dir + "models/resnet_standard_cifar10.pt"))
            model = base
        case ModelTypes.ALEXNET:
            base = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
            base.classifier[6] = nn.Linear(base.classifier[6].in_features, 10)

            base.load_state_dict(
                load(curr_dir + "models/alexnet_standard_cifar10.pt"))
            model = base
        case ModelTypes.DENSENET:
            base = models.densenet121(
                weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            base.classifier = nn.Linear(base.classifier.in_features, 10)

            base.load_state_dict(
                load(curr_dir + "models/densenet_standard_cifar10.pt"))
            model = base
        case ModelTypes.EFFICIENTNET:
            base = hub.load('NVIDIA/DeepLearningExamples:torchhub',
                            'nvidia_efficientnet_b0', pretrained=True)
            base.classifier.fc = nn.Linear(base.classifier.fc.in_features, 10)

            base.load_state_dict(
                load(curr_dir + "models/efficientnet_standard_cifar10.pt"))
            model = base
        case ModelTypes.GOOGLENET:
            base = models.googlenet(
                weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
            base.fc = nn.Linear(base.fc.in_features, 10)

            base.load_state_dict(
                load(curr_dir + "models/googlenet_standard_cifar10.pt"))
            model = base
        case ModelTypes.MOBILENET:
            base = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            base.classifier[1] = nn.Linear(base.classifier[1].in_features, 10)

            base.load_state_dict(
                load(curr_dir + "models/mobilenet_standard_cifar10.pt"))
            model = base
        case ModelTypes.SQUEEZENET:
            base = models.squeezenet1_0(
                weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
            base.classifier[1] = nn.Conv2d(
                512, 10, kernel_size=(1, 1), stride=(1, 1))

            base.load_state_dict(
                load(curr_dir + "models/squeezenet_standard_cifar10.pt"))
            model = base
        case _:
            raise ValueError("Unknown model choice")
    return model


def get_layer_by_name(model, layer_name):
    """
    Gets a layer given a name and model. Raises ValueError if layer is not 
    found in the model.
    :param model: Model to look for layer in
    :param layer_name: Layer name to search
    :return: Layer found
    """
    current_layer = model
    layer_names = layer_name.split('_')  # Split into sub layers

    for name in layer_names:
        if isinstance(current_layer, nn.Module):
            current_layer = getattr(current_layer, name, None)
        else:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")

        if current_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")

    return current_layer


def get_feature_map_sizes(model, layers, img=None):
    """
    Gets the feature map sizes, used to dynamically limit values on the 
    interface.
    :param img: Image to use for forward pass
    :param model: Model to pass image through
    :param layers: Layers to grab feature map sizes from
    :return: Feature map sizes
    """
    feature_map_sizes = [None] * len(layers)

    if img is None:
        img = h_manipulation.create_random_image((227, 227),
            h_manipulation.DatasetNormalizations.CIFAR10_MEAN.value,
            h_manipulation.DatasetNormalizations.CIFAR10_STD.value).clone()
    else:
        img = img.unsqueeze(0)

    # Use GPU if possible
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print("Using GPU for generation")
        model = model.cuda()
        img = img.cuda()
    else:
        print("Using CPU for generation")
    model = model.eval()

    # Fake forward pass for activations
    index = 0
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            hook = layer.register_forward_hook(_get_activations())
            model(img)
            # Activations will have feature map sizes
            feature_map_sizes[index] = _get_activation_shape()
            hook.remove()
        index += 1
    return feature_map_sizes
