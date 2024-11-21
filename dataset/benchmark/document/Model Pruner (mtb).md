- `Model Pruner (mtb)`: The MTB_ModelPruner node is designed for optimizing and pruning machine learning models to enhance performance and efficiency. It supports operations such as precision conversion, removal of unnecessary components, and conditional execution based on model characteristics.
    - Inputs:
        - `save_separately` (Required): Determines whether model components should be saved separately, affecting the organization of the output files. Type should be `BOOLEAN`.
        - `save_folder` (Required): Specifies the directory where the pruned model and its components will be saved. Type should be `STRING`.
        - `fix_clip` (Required): Indicates whether to apply fixes to the CLIP model component, potentially improving compatibility or performance. Type should be `BOOLEAN`.
        - `remove_junk` (Required): Controls the removal of unnecessary or redundant parts of the model to streamline and optimize. Type should be `BOOLEAN`.
        - `ema_mode` (Required): Defines the mode for Exponential Moving Average (EMA) handling within the model, affecting model stability and performance. Type should be `COMBO[STRING]`.
        - `precision_unet` (Required): Sets the precision level for the U-Net model component, impacting memory usage and computational efficiency. Type should be `COMBO[STRING]`.
        - `operation_unet` (Required): Specifies the operation to be performed on the U-Net model component, such as pruning or optimization. Type should be `COMBO[STRING]`.
        - `precision_clip` (Required): Sets the precision level for the CLIP model component, impacting memory usage and computational efficiency. Type should be `COMBO[STRING]`.
        - `operation_clip` (Required): Specifies the operation to be performed on the CLIP model component, such as pruning or optimization. Type should be `COMBO[STRING]`.
        - `precision_vae` (Required): Sets the precision level for the VAE model component, impacting memory usage and computational efficiency. Type should be `COMBO[STRING]`.
        - `operation_vae` (Required): Specifies the operation to be performed on the VAE model component, such as pruning or optimization. Type should be `COMBO[STRING]`.
        - `unet` (Optional): Optional U-Net model component to be pruned or optimized, provided as a dictionary of tensors. Type should be `MODEL`.
        - `clip` (Optional): Optional CLIP model component to be pruned or optimized, provided as a dictionary of tensors. Type should be `CLIP`.
        - `vae` (Optional): Optional VAE model component to be pruned or optimized, provided as a dictionary of tensors. Type should be `VAE`.
    - Outputs: