import torch
import numpy as np
import gradio as gr
import helpers.models as h_models
import lucent.optvis.param as param
import lucent.optvis.transform as tr
import helpers.manipulation as h_manip
import lucent.optvis.objectives as objs

from torch import nn
from time import sleep
from lucent.optvis import render
from lucent.modelzoo.util import get_model_layers


# Event listener functions
def on_model(model, model_layers, ft_map_sizes, evt: gr.SelectData, progress=gr.Progress()):
    """
    Logic flow when model is selected. Updates model, the model layers, and the
    feature map sizes.
    :param model: Current model (object) selected. Updated by this method
    :param model_layers: List of model layers. Updated by this method
    :param ft_map_sizes: List of Feature map sizes. Updated by this method
    :param evt: Event data from Dropdown selection
    :return: [Layer Dropdown Component, Model state, Model Layers state, 
              Feature Map Sizes State]
    """
    progress(0, desc="Setting up model...")
    model = h_models.setup_model(h_models.ModelTypes[evt.value])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    progress(0.25, desc="Getting layers names and details...")
    model_layers = list(get_model_layers(model, 
                                            getLayerRepr=True).items())
    choices = [f"({k}): {v.split('(')[0]}" for k, v in model_layers]


    progress(0.5, desc="Getting layer objects...")
    for i in range(len(model_layers)):
        try:
            layer = h_models.get_layer_by_name(model, model_layers[i][0])
        except ValueError as e:
            gr.Error(e)

        model_layers[i] = (model_layers[i][0], layer)

    progress(0.75, desc="Getting feature maps sizes...")
    ft_map_sizes = h_models.get_feature_map_sizes(model, [v for _, v in model_layers])
    progress(1, desc="Done")
    sleep(0.25)  # To allow for progress animation, not good practice
    return [gr.update(choices=choices, value=''),
            model, model_layers, ft_map_sizes]


def on_layer(selected_layer, model_layers, ft_map_sizes, evt: gr.SelectData):
    """
    Logic flow when a layer is selected. Updates max values of layer
    specific input fields.
    :param selected_layer: Current selected layer, updated by this method.
    :param model_layers: All model layers
    :param ft_map_sizes: Feature maps sizes for all conv layers
    :param evt: Event data from Dropdown selection
    :return [Layer Text Component,
             Channel Number Component,
             Node X Number Component,
             Node Y Number Component,
             Selected layer state/variable,
             Channel max state/variable,
             NodeX max state/variable,
             NodeY max state/variable,
             Node max state/variable]
    """
    channel_max, nodeX_max, nodeY_max, node_max = -1, -1, -1, -1
    selected_layer = model_layers[evt.index]
    match type(selected_layer[1]):
        case nn.Conv2d:
            # Calculate maxes for conv specific
            channel_max = selected_layer[1].out_channels-1
            nodeX_max = ft_map_sizes[evt.index][1]-1
            nodeY_max = ft_map_sizes[evt.index][2]-1
            
            return [gr.update(visible=True),
                    gr.Number.update(info=f"""Values between 0-{channel_max}""", 
                              visible=True, value=None),
                    gr.Number.update(info=f"""Values between 0-{nodeX_max}""", 
                              visible=True, value=None),
                    gr.Number.update(info=f"""Values between 0-{nodeY_max}""", 
                              visible=True, value=None),
                    gr.update(visible=False, value=None),
                    selected_layer,
                    channel_max,
                    nodeX_max,
                    nodeY_max,
                    node_max]
        case nn.Linear:
            # Calculate maxes for linear specific
            node_max = selected_layer[1].out_features-1
            return [gr.update(visible=True),
                    gr.Number.update(visible=False, value=None),
                    gr.Number.update(visible=False, value=None),
                    gr.Number.update(visible=False, value=None),
                    gr.update(info=f"""Values between 0-{node_max}""", 
                              maximum=node_max, 
                              visible=True, value=None),
                    selected_layer,
                    channel_max,
                    nodeX_max,
                    nodeY_max,
                    node_max]
        case _:
            gr.Warning("Unknown layer type")
            return [gr.update(visible=False),
                    gr.update(visible=False, value=None),
                    gr.update(visible=False, value=None),
                    gr.update(visible=False, value=None),
                    gr.update(visible=False, value=None),
                    selected_layer,
                    channel_max,
                    nodeX_max,
                    nodeY_max,
                    node_max]

# Having this many inputs is typically not good practice
def generate(lr, epochs, img_size, channel, nodeX, nodeY, node, layer_sel, 
             model, thresholds, chan_decor, spacial_decor, sd_num, 
             transforms_selected, pad_num, pad_mode, constant_num, jitter_num, 
             scale_num, rotate_num, ad_jitter_num, 
             progress=gr.Progress(track_tqdm=True)):
    """
    Generates the feature visualizaiton with given parameters and tuning. 
    Utilizes the Lucent (Pytorch Lucid library).

    Inputs are different gradio components. Outputs an image component, and 
    their respective epoch numbers. Method tracks its own tqdm progress.
    """
    
    # Image setup
    def param_f(): return param.image(img_size, 
                                      fft=spacial_decor, 
                                      decorrelate=chan_decor,
                                      sd=sd_num)
    
    def optimizer(params): return torch.optim.Adam(params, lr=lr)
    
    # Tranforms setup
    tr_states = {
        h_models.TransformTypes.PAD.value: None,
        h_models.TransformTypes.JITTER.value: None,
        h_models.TransformTypes.RANDOM_SCALE.value: None,
        h_models.TransformTypes.RANDOM_ROTATE.value: None,
        h_models.TransformTypes.AD_JITTER.value: None
    }

    for tr_sel in transforms_selected:
        match tr_sel:
            case h_models.TransformTypes.PAD.value:
                tr_states[tr_sel] = tr.pad(pad_num, 
                                           mode = "constant" if pad_mode == "Constant" else "reflect",
                                           constant_value=constant_num)
            case h_models.TransformTypes.JITTER.value:
                tr_states[tr_sel] = tr.jitter(jitter_num)
            case h_models.TransformTypes.RANDOM_SCALE.value:
                tr_states[tr_sel] = tr.random_scale([1.0 - scale_num + i * (scale_num*2/(51-1)) for i in range(51)])
            case h_models.TransformTypes.RANDOM_ROTATE.value:
                tr_states[tr_sel] = tr.random_rotate([0 - rotate_num + i for i in range(rotate_num*2+1)])
            case h_models.TransformTypes.AD_JITTER.value:
                tr_states[tr_sel] = tr.jitter(ad_jitter_num)
    
    transforms = [t for t in tr_states.values() if t is not None]

    # Specific layer type handling
    match type(layer_sel[1]):
        case nn.Conv2d:
            if (channel is not None and nodeX is not None and nodeY is not None):
                gr.Info("Convolutional Node Specific")
                obj = objs.neuron(layer_sel[0], channel, x=nodeX, y=nodeY)

            elif (channel is not None):
                gr.Info("Convolutional Channel Specific ")
                obj = objs.channel(layer_sel[0], channel)

            elif (channel is None and nodeX is None and nodeY is None):
                gr.Info("Convolutional Layer Specific")
                obj = lambda m: torch.mean(torch.pow(-m(layer_sel[0]).cuda(), 
                                                     torch.tensor(2).cuda())).cuda()
            
            # Unknown
            else:
                gr.Error("Invalid layer settings")
                return None

        case nn.Linear:
            if (node is not None):
                gr.Info("Linear Node Specific")
                obj = objs.channel(layer_sel[0], node)
            else:
                gr.Info("Linear Layer Specific")
                obj = lambda m: torch.mean(torch.pow(-m(layer_sel[0]).cuda(), torch.tensor(2).cuda())).cuda()
        case _:
            gr.Info("Attempting unknown Layer Specific")
            transforms = [] # Just in case
            obj = lambda m: torch.mean(torch.pow(-m(layer_sel[0]).cuda(), torch.tensor(2).cuda())).cuda()

    thresholds = h_manip.expo_tuple(epochs, 6)

    img = np.array(render.render_vis(model,
                                     obj,
                                     thresholds=thresholds,
                                     show_image=False,
                                     optimizer=optimizer,
                                     param_f=param_f,
                                     transforms=transforms,
                                     verbose=True)).squeeze(1)
    
    return gr.Gallery.update(img), thresholds


def update_img_label(epoch_nums, evt: gr.SelectData):
    """
    Updates the image label with its respective epoch number.
    :param epoch_nums: The epoch numbers
    :param evt: Event data from Gallery selection
    :return: Image Gallery Component
    """
    return gr.Gallery.update(label='Epoch ' + str(epoch_nums[evt.index]), 
                             show_label=True)


def check_input(curr, maxx):
    """
    Checks if the current input is higher then the max. Will raise if an error
    if so.
    :param curr: Current value
    :param maxx: Max value to check against
    """
    if curr > maxx:
        raise gr.Error(f"""Value {curr} is higher then maximum of {maxx}""")


def on_transform(transforms):
    """
    Logic for when a transform is selected. Controls the visbility of the
    transform specific inputs/settings.
    :param transforms: The transforms currently selected
    :return: Column Components with modified visibility
    """
    transform_states = {
        h_models.TransformTypes.PAD.value: False,
        h_models.TransformTypes.JITTER.value: False,
        h_models.TransformTypes.RANDOM_SCALE.value: False,
        h_models.TransformTypes.RANDOM_ROTATE.value: False,
        h_models.TransformTypes.AD_JITTER.value: False
    }
    for transform in transforms:
        transform_states[transform] = True

    return [gr.update(visible=state) for state in transform_states.values()]


def on_pad_mode (evt: gr.SelectData):
    """
    Hides the constant value input if the constant pad mode is not selected
    :param evt: Event data from Radio selection
    """
    if (evt.value == "Constant"):
        return gr.update(visible=True)
    return gr.update(visible=False)