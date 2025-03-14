- `YellowBus`: YellowBus is designed to facilitate the transfer of various types of data, including models, VAEs, conditioning prompts, and latent embeddings, without modification. It acts as a conduit, allowing these elements to flow seamlessly from input to output, thereby enabling the preservation and straightforward exchange of information between different stages of a processing pipeline.
    - Inputs:
        - `model` (Required): Represents a model input, allowing for the transfer of model data through the node. Type should be `MODEL`.
        - `vae` (Required): Represents a VAE (Variational Autoencoder) input, facilitating the transfer of VAE data through the node. Type should be `VAE`.
        - `pos_prompt` (Required): Represents a positive conditioning prompt, enabling the transfer of specific conditioning information intended to positively influence the outcome. Type should be `CONDITIONING`.
        - `neg_prompt` (Required): Represents a negative conditioning prompt, enabling the transfer of specific conditioning information intended to negatively influence the outcome. Type should be `CONDITIONING`.
        - `latent` (Required): Represents latent embeddings input, allowing for the transfer of latent space representations through the node. Type should be `LATENT`.
    - Outputs:
        - `model`: Outputs the model data received as input, unchanged. Type should be `MODEL`.
        - `vae`: Outputs the VAE (Variational Autoencoder) data received as input, unchanged. Type should be `VAE`.
        - `pos_prompt`: Outputs the positive conditioning prompt received as input, unchanged. Type should be `CONDITIONING`.
        - `neg_prompt`: Outputs the negative conditioning prompt received as input, unchanged. Type should be `CONDITIONING`.
        - `latent`: Outputs the latent embeddings received as input, unchanged. Type should be `LATENT`.
