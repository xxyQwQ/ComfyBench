- `easy XYInputs_ ModelMergeBlocks`: This node facilitates the merging of models by providing an easy-to-use interface for specifying the models to be merged and the parameters governing the merge process. It abstracts the complexities involved in model merging, making it accessible for users to combine different models to achieve enhanced or specific functionalities.
    - Inputs:
        - `ckpt_name_i` (Required): Specifies the checkpoint name for the subsequent models to be merged. This parameter allows users to select additional model versions or states to combine, enabling the creation of a new, merged model. Type should be `COMBO[STRING]`.
        - `vae_use` (Required): Determines which model's VAE (Variational Autoencoder) is to be used in the merged model or allows for the selection of a specific VAE from available options. Type should be `COMBO[STRING]`.
        - `preset` (Required): Allows users to select a preset configuration for the model merging process, simplifying the setup by providing predefined parameter values. Type should be `COMBO[STRING]`.
        - `values` (Required): Defines the specific values for the merge parameters, such as input, middle, and output block weights, allowing for detailed customization of the merge process. Type should be `STRING`.
    - Outputs:
        - `X or Y`: The output is a merged model, which could be either model X or Y depending on the merging parameters and conditions specified by the user. Type should be `X_Y`.
