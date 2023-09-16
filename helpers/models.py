from enum import Enum
import os
from torch import nn, load, hub
import torch
from torchvision import models

class ModelTypes(Enum):
    RESNET = 0
    ALEXNET = 1
    DENSENET = 2
    EFFICIENTNET = 3
    GOOGLENET = 4
    MOBILENET = 5
    SQUEEZENET = 6

class LayerTypes(Enum):
    CONVOLUTIONAL = nn.Conv2d
    LINEAR = nn.Linear

hook_activations = None

# TODO, possibly expand this to also take in dataset?
def setup_model(model):
    """
    Takes in a model type and creates a standard and robust model. Currently only setups up
    for the CIFAR10 dataset. Loads the respective standard and robust checkpoints saved in the
    models directory.

    :param model: Expects a enum of type ModelTypes to specified model to setup
    :return: model - The returned models tuple (Standard, Robust)
    """
    model_standard = None
    curr_dir = os.path.dirname(__file__) + "/../"
    match model:
        case ModelTypes.RESNET:
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            base.fc = nn.Linear(base.fc.in_features, 10)

            base.load_state_dict(load(curr_dir + "models/resnet_standard_cifar10.pt"))  # Standard
            model_standard = base
        case ModelTypes.ALEXNET:
            base = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
            base.classifier[6] = nn.Linear(base.classifier[6].in_features, 10)

            base.load_state_dict(load(curr_dir + "models/alexnet_standard_cifar10.pt"))  # Standard
            model_standard = base
        case ModelTypes.DENSENET:
            base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            base.classifier = nn.Linear(base.classifier.in_features, 10)

            base.load_state_dict(load(curr_dir + "models/densenet_standard_cifar10.pt"))  # Standard
            model_standard = base
        case ModelTypes.EFFICIENTNET:
            base = hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
            base.classifier.fc = nn.Linear(base.classifier.fc.in_features, 10)

            base.load_state_dict(load(curr_dir + "models/efficientnet_standard_cifar10.pt"))  # Standard
            model_standard = base
        case ModelTypes.GOOGLENET:
            base = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
            base.fc = nn.Linear(base.fc.in_features, 10)

            base.load_state_dict(load(curr_dir + "models/googlenet_standard_cifar10.pt"))  # Standard
            model_standard = base
        case ModelTypes.MOBILENET:
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            base.classifier[1] = nn.Linear(base.classifier[1].in_features, 10)

            base.load_state_dict(load(curr_dir + "models/mobilenet_standard_cifar10.pt"))  # Standard
            model_standard = base
        case ModelTypes.SQUEEZENET:
            base = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
            base.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))

            base.load_state_dict(load(curr_dir + "models/squeezenet_standard_cifar10.pt"))  # Standard
            model_standard = base
        case _:
            print("Unknown model choice")
            exit()
    return model_standard

def get_instance_type_paths(model, instance_type):
    """
    Recursively traverses model to look for specific instance types.
    If is the correct instance type, saves the full path to the instance.
    :param model: Model to dig through
    :param instance_type: Instance(s) to save paths for (such as nn.Conv2d)
    :return: list of all the instance type paths
    """
    paths = []
    for name, module in model.named_children():
        if isinstance(module, instance_type):
            # Base case: if module is of correct instance type, return its name
            paths.append(name)
        elif len(list(module.children())) > 0:
            # Recursive case: if module has children, recursively call get_instance_type_paths on each child
            child_paths = get_instance_type_paths(module, instance_type)
            for child_path in child_paths:
                paths.append(name + '.' + child_path)
        else:
            # Some layers are unnamed, so we need to handle them explicitly
            if isinstance(module, nn.Sequential) and isinstance(module[0], instance_type):
                paths.append(name + '.0')
    return paths

# Hook function
def get_activations():
    """
    Used when registering forward hooks to get layer activations
    :return:
    """

    def hook(model, input, output):
        global hook_activations
        hook_activations = output

    return hook

def get_activation_shape():
    return hook_activations.squeeze().size()

def get_feature_map_sizes(img, model, layers):
    feature_map_sizes = [None] * len(layers)

    # Preprocess
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

    index=0
    for layer in layers:
        hook = layer.register_forward_hook(get_activations())
        model(img)
        feature_map_sizes[index] = get_activation_shape()
        index+=1
        hook.remove()
    
    return feature_map_sizes




def get_all_instances(model, instance_type):
    """
    Gets all the instance_type objects through out a model. Such as getting all of the
    Conv2d objects of a model.
    :param model: THe model to search through
    :param instance_type: The instance type(s) to look for
    :return: List of all objects of instance_type
    """
    instance_paths = get_instance_type_paths(model, instance_type)  # Get all paths to instance types
    layers = [None] * len(instance_paths)  # Static allocation as we know how big it needs to be
    index = 0

    # # Collect the actual isntance objects
    for path in instance_paths:
        layer = model
        for attr_name in path.split('.'):
            layer = getattr(layer, attr_name)
        layers[index] = layer
        index += 1

    return layers, instance_paths