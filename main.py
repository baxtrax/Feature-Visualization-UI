import torch
import numpy as np
import gradio as gr
import helpers.models as h_models
import helpers.manipulation as h_manip
import lucent.optvis.param as param
import lucent.optvis.objectives as objectives
from torch import nn
from time import sleep
from lucent.optvis import render
from lucent.modelzoo.util import get_model_layers


deep_orange = gr.themes.Color(c50="#FFEDE5",
                              c100="#FFDACC",
                              c200="#FFB699",
                              c300="#FF9166",
                              c400="#FF6D33",
                              c500="#FF4700",
                              c600="#CC3A00",
                              c700="#992B00",
                              c800="#661D00",
                              c900="#330E00",
                              c950="#190700")
css = """
div[data-testid="block-label"] {z-index: var(--layer-3)}
"""

def main():
    # with gr.Blocks(theme=gr.themes.Soft(primary_hue=deep_orange,
    #                                     secondary_hue=deep_orange,
    #                                     neutral_hue=gr.themes.colors.zinc)) as demo:
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        # Session states
        selected_layer = gr.State(None)
        model, model_layers = gr.State(None), gr.State(None)
        ft_map_sizes = gr.State(None)
        thresholds = gr.State(None)

        # GUI Elements
        with gr.Row():  # Upper banner
            gr.Markdown("""# Feature Visualization Generator\n
                             Start by selecting a model from the drop down.""")
        with gr.Row():  # Lower inputs and outputs
            with gr.Column():  # Inputs
                gr.Markdown("""## Model Settings""")

                model_dd = gr.Dropdown(label="Model",
                                       info="Select a model. Some models take longer to setup",
                                       choices=[m.name for m in h_models.ModelTypes])

                layer_dd = gr.Dropdown(label="Layer",
                                       info="Select a layer. List will change depending on layer type selected",
                                       interactive=True,
                                       visible=False)

                layer_text = gr.Markdown("""## Layer Settings (Optional)""",
                                         visible=False)

                with gr.Row():  # Inputs specific to layer selection
                    channel_num = gr.Number(label="Channel",
                                            info="Please choose a layer",
                                            precision=0,
                                            minimum=0,
                                            maximum=100,
                                            interactive=True,
                                            visible=False,
                                            value=None)

                    node_num = gr.Number(label="Node",
                                         info="Please choose a layer",
                                         precision=0,
                                         minimum=0,
                                         maximum=10,
                                         interactive=True,
                                         visible=False,
                                         value=None)

                    nodeX_num = gr.Number(label="Node X",
                                          info="Please choose a layer",
                                          precision=0,
                                          minimum=0,
                                          maximum=64,
                                          interactive=True,
                                          visible=False,
                                          value=None)

                    nodeY_num = gr.Number(label="Node Y",
                                          info="Please choose a layer",
                                          precision=0,
                                          minimum=0,
                                          maximum=64,
                                          interactive=True,
                                          visible=False,
                                          value=None)

                gr.Markdown("""## Visualization Settings""")

                lr_sl = gr.Slider(label="Learning Rate",
                                  info="How aggresive each \"step\" towards the visualization is",
                                  minimum=0.000001,
                                  maximum=3,
                                  step=0.000001,
                                  value=0.125)

                epoch_num = gr.Number(label="Epochs",
                                      info="How many steps (epochs) to perform",
                                      precision=0,
                                      minimum=1,
                                      value=200)

                img_num = gr.Number(label="Image Size",
                                    info="Image is square (<value> by <value>)",
                                    precision=0,
                                    minimum=1,
                                    value=227)
                
                with gr.Accordion("Advanced Settings", open=False):
                    gr.Markdown("""## Image Settings (WIP)""")
                    gr.Checkbox(label="Decorrelate Image", info="Only works if channels # are unspecified")
                    gr.Checkbox(label="FFT")
                    gr.Number(label="Channels", info="Defaults to 3 if unspecified")
                    gr.Number(label="Batch", info="Defaults to 1 if unspecified")

                    gr.Markdown("""## Transform Settings (WIP)""")
                    gr.Checkbox(label="Preprocess", info="Enable or disable preprocessing via transformations")
                    gr.Dropdown(label="Applied Transforms", info="Transforms to apply", multiselect=True)

                confirm_btn = gr.Button("Generate", visible=False)

            with gr.Column(): # Output
                gr.Markdown("""## Feature Visualization Output""")
                with gr.Row():
                    images_gal = gr.Gallery(show_label=False,
                                            preview=True,
                                            allow_preview=True)

        # Event listener binding
        model_dd.select(lambda: gr.Dropdown.update(visible=True),
                        outputs=layer_dd)
        model_dd.select(on_model, 
                        inputs=[model, model_layers, ft_map_sizes], 
                        outputs=[layer_dd, model, model_layers, ft_map_sizes])

        # TODO: Make button invisible always until layer selection
        layer_dd.select(lambda: gr.Button.update(visible=True),
                        outputs=confirm_btn)
        layer_dd.select(on_layer, 
                        inputs=[selected_layer, model_layers, ft_map_sizes],
                        outputs=[layer_text,
                                 channel_num,
                                 nodeX_num,
                                 nodeY_num,
                                 node_num,
                                 selected_layer])

        confirm_btn.click(generate,
                          inputs=[lr_sl,
                                  epoch_num,
                                  img_num,
                                  channel_num,
                                  nodeX_num,
                                  nodeY_num,
                                  node_num, 
                                  selected_layer, 
                                  model, 
                                  thresholds],
                          outputs=[images_gal, thresholds])
        images_gal.select(update_img_label,
                          inputs=thresholds,
                          outputs=images_gal)

    demo.queue().launch()


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
             Selected layer state/variable]
    """
    selected_layer = model_layers[evt.index]
    match type(selected_layer[1]):
        case nn.Conv2d:
            channel_max = selected_layer[1].out_channels-1
            nodeX_max = ft_map_sizes[evt.index][1]-1
            nodeY_max = ft_map_sizes[evt.index][2]-1
            return [gr.update(visible=True),
                    gr.update(info=f"""Values between 0-{channel_max}""", 
                              maximum=channel_max, 
                              visible=True, value=None),
                    gr.update(info=f"""Values between 0-{nodeX_max}""", 
                              maximum=nodeX_max, 
                              visible=True, value=None),
                    gr.update(info=f"""Values between 0-{nodeY_max}""", 
                              maximum=nodeY_max, 
                              visible=True, value=None),
                    gr.update(visible=False, value=None),
                    selected_layer]
        case nn.Linear:
            node_max = selected_layer[1].out_features-1
            return [gr.update(visible=True),
                    gr.update(visible=False, value=None),
                    gr.update(visible=False, value=None),
                    gr.update(visible=False, value=None),
                    gr.update(info=f"""Values between 0-{node_max}""", 
                              maximum=node_max, 
                              visible=True, value=None),
                    selected_layer]
        case _:
            gr.Warning("Unknown layer type")
            return [gr.update(visible=False),
                    gr.update(visible=False, value=None),
                    gr.update(visible=False, value=None),
                    gr.update(visible=False, value=None),
                    gr.update(visible=False, value=None),
                    selected_layer]


def generate(lr, epochs, img_size, channel, nodeX, nodeY, node, selected_layer, 
             model, thresholds, progress=gr.Progress(track_tqdm=True)):
    """
    Generates the feature visualizaiton with given parameters and tuning. 
    Utilizes the Lucent (Pytorch Lucid library).

    Inputs are different gradio components. Outputs an image component. Method 
    tracks its own tqdm progress.
    """
    def param_f(): return param.image(img_size)  # Image setup
    def optimizer(params): return torch.optim.Adam(params, lr=lr)

    # Specific layer type handling
    match type(selected_layer[1]):
        case nn.Conv2d:
            # Node specific
            if (channel is not None and nodeX is not None and nodeY is not None):
                gr.Info("Node Specific Convolution")
                obj = objectives.neuron(selected_layer[0],
                                        channel,
                                        x=nodeX,
                                        y=nodeY)

            # Channel specific
            elif (channel is not None):
                gr.Info("Channel Specific Convolution")
                obj = objectives.channel(selected_layer[0], channel)

            # Layer specific
            elif (channel is None and nodeX is None and nodeY is None):
                gr.Info("Layer Specific Convolution")
                obj = lambda m: torch.mean(torch.pow(-m(selected_layer[0]).cuda(), torch.tensor(2).cuda())).cuda()
            
            # Unknown
            else:
                gr.Error("Invalid layer settings")
                return None

        case nn.Linear:
            if (node is not None):  # Node Specific
                obj = objectives.channel(selected_layer[0], node)
            else:  # Layer Specific
                obj = lambda m: torch.mean(torch.pow(-m(selected_layer[0]).cuda(), torch.tensor(2).cuda())).cuda()
    thresholds = h_manip.expo_tuple(epochs, 6)
    img = np.array(render.render_vis(model,
                                obj,
                                thresholds=thresholds,
                                show_image=False,
                                optimizer=optimizer,
                                param_f=param_f)).squeeze(1)
    
    return gr.Gallery.update(img), thresholds


def update_img_label(thresholds, evt: gr.SelectData):
    return gr.Gallery.update(label='Epoch ' + str(thresholds[evt.index]), show_label=True)

main()
