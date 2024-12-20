- `LDSRModelLoader`: The LDSRModelLoader node is designed to load and prepare LDSR (Low-Dimensional Super-Resolution) models for use, specifically focusing on selecting and initializing models for image upscaling tasks.
    - Inputs:
        - `model` (Required): Specifies the model to be loaded for the upscaling task. The choice of model can influence the quality and characteristics of the upscaling result. Type should be `COMBO[STRING]`.
    - Outputs:
        - `upscale_model`: Returns the loaded and CPU-transferred LDSR model ready for image upscaling tasks. Type should be `UPSCALE_MODEL`.
