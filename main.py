import gradio as gr
import helpers.models as h_models
import helpers.listeners as listeners

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
                             FVs are a part of a wider field called Explainable 
                             Artificial Intelligence (XAI) This generator aims 
                             to make it easier to explore different concepts 
                             used in FV generation and allow for experimentation.
                             Currently Convolutional and Linear layers were tested.\n\n
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
                                                       precision=0,
                                                       interactive=True)

                        rand_scale_col = gr.Column()
                        with rand_scale_col:
                            gr.Markdown("""### Random Scale Settings""")
                            with gr.Row():
                                scale_num = gr.Number(label="Max scale",
                                                      info="How much to scale (from 1.0) in both directions (+ and -)",
                                                      minimum=0,
                                                      value=0.1,
                                                      interactive=True)
                        
                        rand_rotate_col = gr.Column()
                        with rand_rotate_col:
                            gr.Markdown("""### Random Rotate Settings""")
                            with gr.Row():
                                rotate_num = gr.Number(label="Max angle",
                                                       info="How much to rotate in both directions (+ and -)",
                                                       minimum=0,
                                                       value=10,
                                                       precision=0,
                                                       interactive=True)
                        
                        ad_jitter_col = gr.Column()
                        with ad_jitter_col:
                            gr.Markdown("""### Additional Jitter Settings""")
                            with gr.Row():
                                ad_jitter_num = gr.Number(label="Jitter",
                                                          info="How much to jitter image by",
                                                          minimum=1,
                                                          value=4,
                                                          precision=0,
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
        model_dd.select(listeners.on_model, 
                        inputs=[model, model_layers, ft_map_sizes], 
                        outputs=[layer_dd, model, model_layers, ft_map_sizes])

        # TODO: Make button invisible always until layer selection
        layer_dd.select(lambda: gr.Button.update(visible=True),
                        outputs=confirm_btn)
        layer_dd.select(listeners.on_layer, 
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
        
        channel_num.blur(listeners.check_input, inputs=[channel_num, channel_max])
        nodeX_num.blur(listeners.check_input, inputs=[nodeX_num, nodeX_max])
        nodeY_num.blur(listeners.check_input, inputs=[nodeY_num, nodeY_max])
        node_num.blur(listeners.check_input, inputs=[node_num, node_max])

        images_gal.select(listeners.update_img_label,
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
        
        transforms_dd.change(listeners.on_transform,
                             inputs=transforms_dd,
                             outputs=[pad_col,
                                      jitter_col,
                                      rand_scale_col,
                                      rand_rotate_col,
                                      ad_jitter_col])
        
        mode_rad.select(listeners.on_pad_mode,
                        outputs=constant_num)

        confirm_btn.click(listeners.generate,
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
                                  sd_num,
                                  transforms_dd,
                                  pad_num,
                                  mode_rad,
                                  constant_num,
                                  jitter_num,
                                  scale_num,
                                  rotate_num,
                                  ad_jitter_num],
                          outputs=[images_gal, thresholds])
        
    demo.queue().launch()

main()
