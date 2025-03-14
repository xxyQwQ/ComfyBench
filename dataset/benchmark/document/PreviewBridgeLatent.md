- `PreviewBridgeLatent`: This node is designed to bridge the gap between various latent representations and their visual previews, adapting to different preview methods and ensuring compatibility with specific latent formats. It plays a crucial role in visualizing latent spaces by converting them into interpretable images, accommodating various generative models and preview techniques.
    - Inputs:
        - `latent` (Required): The latent representation to be visualized. It is crucial for determining the visual output and ensuring it aligns with the selected preview method. Type should be `LATENT`.
        - `image` (Required): The image data that may be used or modified in the visualization process, depending on the preview method and the presence of a mask. Type should be `STRING`.
        - `preview_method` (Required): Specifies the method used for visualizing the latent representation. It affects the conversion process and the compatibility with the latent format. Type should be `COMBO[STRING]`.
        - `vae_opt` (Optional): An optional parameter that allows for the specification of a variational autoencoder option, further customizing the visualization process. Type should be `VAE`.
    - Outputs:
        - `latent`: The processed latent representation, potentially modified or enhanced through the visualization process. Type should be `LATENT`.
        - `mask`: An optional mask applied to the latent visualization, indicating areas of interest or modification. Type should be `MASK`.
