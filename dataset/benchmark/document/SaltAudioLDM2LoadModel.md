- `SaltAudioLDM2LoadModel`: This node is designed to load a specific audio latent diffusion model (AudioLDM2) into memory, making it ready for audio processing tasks. It supports loading different versions of the model and allows specifying the computational device (CPU or GPU) for the model's operations.
    - Inputs:
        - `model` (Required): Specifies the version of the AudioLDM2 model to load. The choice of model can significantly impact the quality and characteristics of the generated audio. Type should be `COMBO[STRING]`.
        - `device` (Optional): Determines the computational device ('cuda' for GPU or 'cpu' for CPU) on which the model will be loaded and executed, affecting performance and efficiency. Type should be `COMBO[STRING]`.
    - Outputs:
        - `audioldm2_model`: The loaded AudioLDM2 model, ready for audio processing tasks. Type should be `AUDIOLDM_MODEL`.