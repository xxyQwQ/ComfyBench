- `Checkpoint Loader`: The Checkpoint Loader node is designed for advanced loading operations, specifically to load various components such as models, CLIP, and VAE from specified checkpoints and configuration files. It facilitates the restoration of these components to their saved states, enabling the continuation of work or further manipulation.
    - Parameters:
        - `config_name`: Specifies the name of the configuration file to be used for loading. It plays a crucial role in determining the setup and parameters of the components to be loaded. Type should be `COMBO[STRING]`.
        - `ckpt_name`: Indicates the name of the checkpoint file from which the components are to be loaded. This file contains the saved state of the components. Type should be `COMBO[STRING]`.
    - Inputs:
    - Outputs:
        - `MODEL`: The loaded model component. Type should be `MODEL`.
        - `CLIP`: The loaded CLIP component. Type should be `CLIP`.
        - `VAE`: The loaded VAE component. Type should be `VAE`.
        - `NAME_STRING`: The name string derived from the checkpoint file, providing an identifier for the loaded components. Type should be `STRING`.