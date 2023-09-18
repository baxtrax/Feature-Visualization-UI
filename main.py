import torch
import numpy as np
import gradio as gr
import helpers.models as h_models
import lucent.optvis.param as param
import helpers.manipulation as h_manipulation
import lucent.optvis.objectives as objectives
from torch import nn
from time import sleep
from lucent.optvis import render
from lucent.modelzoo.util import get_model_layers


custom_color = gr.themes.Color(c50="#FFEDE5",
                               c100="#FFDACC",
                               c200="#FFB699",
                               c300="#FF9166",
                               c400="#FF6D33",
                               c500="#FF4700",
                               c600="#CC3A00",
                               c700="#992B00",
                               c800="#661D00",
                               c900="#330E00",
                               c950="#190700",)

image_block = None
selected_layer = None
model, model_layers, layer_feature_map_sizes = None, None, None


def main():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue=custom_color,
                                        secondary_hue=custom_color,
                                        neutral_hue=gr.themes.colors.zinc)) as demo:
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
                                  value=0.01)

                epoch_num = gr.Number(label="Epochs",
                                      info="How many steps (epochs) to perform",
                                      precision=0,
                                      minimum=1,
                                      value=200)

                image_num = gr.Number(label="Image Size",
                                      info="Image is square (<value> by <value>)",
                                      precision=0,
                                      minimum=1,
                                      value=227)

                confirm_btn = gr.Button("Generate", visible=False)

            with gr.Column(): # Output
                image_block = gr.Image(label="Output",
                                       value=None,
                                       interactive=False)

        # Event listener functions
        def on_model(evt: gr.SelectData, progress=gr.Progress()):
            """
            Logic flow when model is selected
            """
            progress(0, desc="Setting up model...")
            global model
            model = h_models.setup_model(h_models.ModelTypes[evt.value])
            progress(0.25, desc="Getting layers names and details...")

            # Layer agnostic logic
            global model_layers
            model_layers = list(get_model_layers(
                model, getLayerRepr=True).items())
            # TODO Use F strings
            choices = ['('+k+'): '+v.split('(')[0] for k, v in model_layers]

            progress(0.5, desc="Getting layer objects...")

            for i in range(len(model_layers)):
                # TODO Error catching here for layer name
                layer = h_models.get_layer_by_name(model, model_layers[i][0])
                model_layers[i] = (model_layers[i][0], layer)

            progress(0.75, desc="Getting feature maps sizes...")
            global layer_feature_map_sizes  # TODO DONT NEED RANDOM IMAGE
            layer_feature_map_sizes = h_models.get_feature_map_sizes(h_manipulation.create_random_image((227, 227),
                                                                                                        h_manipulation.DatasetNormalizations.CIFAR10_MEAN.value,
                                                                                                        h_manipulation.DatasetNormalizations.CIFAR10_STD.value).clone(),
                                                                     model,
                                                                     [v for k, v in model_layers])
            progress(1, desc="Done")
            sleep(0.25)  # To allow for progress animation
            return gr.update(choices=choices, value=None)

        def on_layer(evt: gr.SelectData):
            """
            Logic flow when a layer is selected
            """
            global selected_layer
            selected_layer = model_layers[evt.index]
            match type(selected_layer[1]):
                case nn.Conv2d:
                    channel_num.maximum = selected_layer[1].out_channels-1
                    nodeX_num.maximum = layer_feature_map_sizes[evt.index][1]-1
                    nodeY_num.maximum = layer_feature_map_sizes[evt.index][2]-1
                    return {layer_text:  gr.update(visible=True),
                            channel_num: gr.update(info=f"""Values between {channel_num.minimum}-{channel_num.maximum}""", 
                                                   visible=True, value=None),
                            nodeX_num:   gr.update(info=f"""Values between {nodeX_num.minimum}-{nodeX_num.maximum}""", 
                                                   visible=True, value=None),
                            nodeY_num:   gr.update(info=f"""Values between {nodeY_num.minimum}-{nodeY_num.maximum}""", 
                                                   visible=True, value=None),
                            node_num:    gr.update(visible=False, value=None)}
                case nn.Linear:
                    node_num.maximum = selected_layer[1].out_features-1
                    return {layer_text:  gr.update(visible=True),
                            channel_num: gr.update(visible=False, value=None),
                            nodeX_num:   gr.update(visible=False, value=None),
                            nodeY_num:   gr.update(visible=False, value=None),
                            node_num:    gr.update(info=f"""Values between {node_num.minimum}-{node_num.maximum}""", 
                                                   visible=True, value=None)}
                case _:
                    gr.Warning("Unknown layer type")
                    return {layer_text:  gr.update(visible=False),
                            channel_num: gr.update(visible=False, value=None),
                            nodeX_num:   gr.update(visible=False, value=None),
                            nodeY_num:   gr.update(visible=False, value=None),
                            node_num:    gr.update(visible=False, value=None)}

        # Event listener binding
        model_dd.select(lambda t: gr.Radio.update(visible=True), 
                        outputs=layer_dd)
        model_dd.select(on_model, outputs=layer_dd)

        # TODO: Make button invisible always until layer selection
        layer_dd.select(lambda t: gr.Button.update(visible=True), 
                        outputs=confirm_btn)
        layer_dd.select(on_layer, 
                        outputs=[layer_text,
                                 channel_num,
                                 nodeX_num,
                                 nodeY_num,
                                 node_num])

        confirm_btn.click(generate,
                          inputs=[lr_sl,
                                  epoch_num,
                                  image_num,
                                  channel_num,
                                  nodeX_num,
                                  nodeY_num,
                                  node_num],
                          outputs=image_block)

    demo.queue().launch()


def generate(lr, epochs, image_num, channel, nodeX, nodeY, node, 
             progress=gr.Progress(track_tqdm=True)):
    """
    Generates the feature visualizaiton with given parameters and tuning. 
    Utilizes the Lucent (Pytorch Lucid library).

    Inputs are different gradio components. Outputs an image component. Method 
    tracks its own tqdm progress.
    """
    def param_f(): return param.image(image_num)  # Image setup
    def optimizer(params): return torch.optim.Adam(params, lr=lr)

    global selected_layer

    # Specific layer type handling
    match type(selected_layer[1]):
        case nn.Conv2d:
            if (channel is not None and  # Node Specific
                nodeX is not None and
                nodeY is not None):
                gr.Info("Node Specific Convolution")
                image = np.array(render.render_vis(model,
                                                   objectives.neuron(selected_layer[0],
                                                                     channel,
                                                                     x=nodeX,
                                                                     y=nodeY),
                                                   thresholds=(epochs, ),
                                                   show_image=False,
                                                   optimizer=optimizer,
                                                   param_f=param_f)).squeeze(0).squeeze(0)

            elif (channel is not None):  # Channel Specific
                gr.Info("Channel Specific Convolution")
                image = np.array(render.render_vis(model,
                                                   objectives.channel(
                                                       selected_layer[0], channel),
                                                   thresholds=(epochs, ),
                                                   show_image=False,
                                                   optimizer=optimizer,
                                                   param_f=param_f)).squeeze(0).squeeze(0)

            elif (channel is None and  # Layer Specific
                  nodeX is None and
                  nodeY is None):
                gr.Info("Layer Specific Convolution")
                image = np.array(render.render_vis(model,
                                                   lambda m: torch.mean(
                                                       torch.pow(-m(selected_layer[0]).cuda(), torch.tensor(2).cuda())).cuda(),
                                                   thresholds=(epochs, ),
                                                   show_image=False,
                                                   optimizer=optimizer,
                                                   param_f=param_f)).squeeze(0).squeeze(0)
            else:
                gr.Error("Invalid layer settings")
                return {image_block: None}

            return {image_block: gr.Image.update(image, True)}

        case nn.Linear:
            if (node is not None):  # Node Specific
                image = np.array(render.render_vis(model,
                                                   objectives.channel(
                                                       selected_layer[0], node),
                                                   thresholds=(epochs, ),
                                                   show_image=False,
                                                   optimizer=optimizer,
                                                   param_f=param_f)).squeeze(0).squeeze(0)
            else:  # Layer Specific
                image = np.array(render.render_vis(model,
                                                   lambda m: torch.mean(
                                                       torch.pow(-m(selected_layer[0]).cuda(), torch.tensor(2).cuda())).cuda(),
                                                   thresholds=(epochs, ),
                                                   show_image=False,
                                                   optimizer=optimizer,
                                                   param_f=param_f)).squeeze(0).squeeze(0)

            return {image_block: gr.Image.update(image, True)}


main()
