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

# Custom css
css = """div[data-testid="block-label"] {z-index: var(--layer-3)}"""

def main():
    with gr.Blocks(title="Feature Visualization Generator", 
                   css=css, 
                   theme=gr.themes.Soft(primary_hue="blue",
                                        secondary_hue="blue",
                                        )) as demo:
        
        # Session state init
        model, model_layers, selected_layer, ft_map_sizes, \
        thresholds, channel_max, nodeX_max, nodeY_max, \
        node_max = (gr.State(None) for _ in range(9))

        # GUI Elements
        with gr.Row():  # Upper banner
            gr.Markdown("""# Feature Visualization Generator\n
                             Feature Visualizations (FV's) answer questions 
                             about what a network—or parts of a network—are 
                             looking for by generating examples. 
                             ([Read more about it here](https://distill.pub/2017/feature-visualization/))
                             This generator aims to make it easier to explore 
                             different concepts used in FV generation and allow 
                             for experimentation.\n\n
                             **Start by selecting a model from the drop down.**""")
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
                                            interactive=True,
                                            visible=False,
                                            value=None)

                    node_num = gr.Number(label="Node",
                                         info="Please choose a layer",
                                         precision=0,
                                         minimum=0,
                                         interactive=True,
                                         visible=False,
                                         value=None)

                    nodeX_num = gr.Number(label="Node X",
                                          info="Please choose a layer",
                                          precision=0,
                                          minimum=0,
                                          interactive=True,
                                          visible=False,
                                          value=None)

                    nodeY_num = gr.Number(label="Node Y",
                                          info="Please choose a layer",
                                          precision=0,
                                          minimum=0,
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
                
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Column(variant="panel"):
                        gr.Markdown("""## Image Settings""")
                        img_num = gr.Number(label="Image Size",
                                            info="Image is square (<value> by <value>)",
                                            precision=0,
                                            minimum=1,
                                            value=227)
                        chan_decor_ck = gr.Checkbox(label="Channel Decorrelation", 
                                                    info="Reduces channel-to-channel correlations",
                                                    value=True)
                        spacial_decor_ck = gr.Checkbox(label="Spacial Decorrelation (FFT)",
                                                       info="Reduces pixel-to-pixel correlations",
                                                       value=True)
                        sd_num = gr.Number(label="Standard Deviation",
                                           info="The STD of the randomly generated starter image",
                                           value=0.01)

                    with gr.Column(variant="panel"):
                        gr.Markdown("""## Transform Settings (WIP)""")
                        preprocess_ck = gr.Checkbox(label="Preprocess",
                                                    info="Enable or disable preprocessing via transformations",
                                                    value=True,
                                                    interactive=True)
                        transform_choices = [t.value for t in h_models.TransformTypes]
                        transforms_dd = gr.Dropdown(label="Applied Transforms", 
                                                    info="Transforms to apply",
                                                    choices=transform_choices,
                                                    multiselect=True,
                                                    value=transform_choices,
                                                    interactive=True)
                        
                        # Transform specific settings
                        pad_col = gr.Column()
                        with pad_col:
                            gr.Markdown("""### Pad Settings""")
                            with gr.Row():
                                pad_num = gr.Number(label="Padding",
                                                    info="How many pixels of padding",
                                                    minimum=0,
                                                    value=12,
                                                    precision=0,
                                                    interactive=True)
                                mode_rad = gr.Radio(label="Mode",
                                                    info="Constant fills padded pixels with a value. Reflect fills with edge pixels",
                                                    choices=["Constant", "Reflect"],
                                                    value="Constant",
                                                    interactive=True)
                                constant_num = gr.Number(label="Constant Fill Value",
                                                         info="Value to fill padded pixels",
                                                         value=0.5,
                                                         interactive=True)

                        jitter_col = gr.Column()
                        with jitter_col:
                            gr.Markdown("""### Jitter Settings""")
                            with gr.Row():
                                jitter_num = gr.Number(label="Jitter",
                                                       info="How much to jitter image by",
                                                       minimum=1,
                                                       value=8,
                                                       interactive=True)

                        rand_scale_col = gr.Column()
                        with rand_scale_col:
                            gr.Markdown("""### Random Scale Settings""")
                            with gr.Row():
                                scale_num = gr.Number(label="Max scale",
                                                      info="How much to scale in both directions (+ and -)",
                                                      minimum=0,
                                                      value=10,
                                                      interactive=True)
                        
                        rand_rotate_col = gr.Column()
                        with rand_rotate_col:
                            gr.Markdown("""### Random Rotate Settings""")
                            with gr.Row():
                                rotate_num = gr.Number(label="Max angle",
                                                       info="How much to rotate in both directions (+ and -)",
                                                       minimum=0,
                                                       value=10,
                                                       interactive=True)
                        
                        ad_jitter_col = gr.Column()
                        with ad_jitter_col:
                            gr.Markdown("""### Additional Jitter Settings""")
                            with gr.Row():
                                ad_jitter_num = gr.Number(label="Jitter",
                                                          info="How much to jitter image by",
                                                          minimum=1,
                                                          value=4,
                                                          interactive=True)
 



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
                                 selected_layer,
                                 channel_max,
                                 nodeX_max,
                                 nodeY_max,
                                 node_max])
        
        channel_num.blur(check_input, inputs=[channel_num, channel_max])
        nodeX_num.blur(check_input, inputs=[nodeX_num, nodeX_max])
        nodeY_num.blur(check_input, inputs=[nodeY_num, nodeY_max])
        node_num.blur(check_input, inputs=[node_num, node_max])

        images_gal.select(update_img_label,
                    inputs=thresholds,
                    outputs=images_gal)
        
        preprocess_ck.select(lambda status: (gr.update(visible=status), 
                                             gr.update(visible=status), 
                                             gr.update(visible=status), 
                                             gr.update(visible=status), 
                                             gr.update(visible=status), 
                                             gr.update(visible=status)),
                             inputs=preprocess_ck,
                             outputs=[transforms_dd,
                                      pad_col,
                                      jitter_col,
                                      rand_scale_col,
                                      rand_rotate_col,
                                      ad_jitter_col])
        
        transforms_dd.change(on_transform,
                             inputs=transforms_dd,
                             outputs=[pad_col,
                                      jitter_col,
                                      rand_scale_col,
                                      rand_rotate_col,
                                      ad_jitter_col])
        
        mode_rad.select(on_pad_mode,
                        outputs=constant_num)

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
                                  thresholds,
                                  chan_decor_ck,
                                  spacial_decor_ck,
                                  sd_num],
                          outputs=[images_gal, thresholds])
        
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
             Selected layer state/variable]
    """
    channel_max, nodeX_max, nodeY_max, node_max = -1, -1, -1, -1
    selected_layer = model_layers[evt.index]
    match type(selected_layer[1]):
        case nn.Conv2d:
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


def generate(lr, epochs, img_size, channel, nodeX, nodeY, node, selected_layer, 
             model, thresholds, chan_decor, spacial_decor, 
             sd_num, progress=gr.Progress(track_tqdm=True)):
    """
    Generates the feature visualizaiton with given parameters and tuning. 
    Utilizes the Lucent (Pytorch Lucid library).

    Inputs are different gradio components. Outputs an image component. Method 
    tracks its own tqdm progress.
    """

    def param_f(): return param.image(img_size, 
                                      fft=spacial_decor, 
                                      decorrelate=chan_decor,
                                      sd=sd_num)  # Image setup
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
    print(thresholds)


    img = np.array(render.render_vis(model,
                                     obj,
                                     thresholds=thresholds,
                                     show_image=False,
                                     optimizer=optimizer,
                                     param_f=param_f,
                                     verbose=True)).squeeze(1)
    
    return gr.Gallery.update(img), thresholds


def update_img_label(thresholds, evt: gr.SelectData):
    return gr.Gallery.update(label='Epoch ' + str(thresholds[evt.index]), show_label=True)


def check_input(curr, maxx):
    if curr > maxx:
        raise gr.Error(f"""Value {curr} is higher then maximum of {maxx}""")


def on_transform(transforms):
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
    if (evt.value == "Constant"):
        return gr.update(visible=True)
    return gr.update(visible=False)
main()
