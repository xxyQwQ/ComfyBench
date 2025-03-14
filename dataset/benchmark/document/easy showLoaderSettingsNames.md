- `easy showLoaderSettingsNames`: This node is designed to display the names of various loader settings, providing a straightforward way for users to understand and interact with the configuration options available for data loading processes.
    - Inputs:
        - `pipe` (Required): Represents the pipeline configuration, crucial for determining the loader settings to be displayed. Type should be `PIPE_LINE`.
        - `names` (Required): Optional parameter allowing the specification of additional information or constraints on the names to be displayed. Type should be `INFO`.
    - Outputs:
        - `ckpt_name`: The name of the checkpoint file as determined by the loader settings. Type should be `STRING`.
        - `vae_name`: The name of the VAE model file as determined by the loader settings. Type should be `STRING`.
        - `lora_name`: The name of the LoRA model file as determined by the loader settings. Type should be `STRING`.
