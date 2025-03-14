- `ADE_InjectI2VIntoAnimateDiffModel`: This node is designed to inject an Image-to-Video (I2V) model into an existing AnimateDiff model, enhancing its capabilities with I2V features. It focuses on integrating specific I2V components into the AnimateDiff framework to enable or improve motion-based animations.
    - Inputs:
        - `model_name` (Required): Specifies the name of the model to be loaded and modified. It is crucial for identifying the correct AnimateDiff model to which the I2V features will be added. Type should be `COMBO[STRING]`.
        - `motion_model` (Required): Represents the motion model patcher object that will be modified to include I2V features. It is essential for applying the I2V enhancements to the specified AnimateDiff model. Type should be `MOTION_MODEL_ADE`.
        - `ad_settings` (Optional): Optional settings for the AnimateDiff model that can be adjusted during the I2V injection process. These settings allow for customization and fine-tuning of the model's behavior. Type should be `AD_SETTINGS`.
        - `deprecation_warning` (Optional): Provides a warning message about the experimental status of this node, indicating that it may not function as expected. This parameter allows for communication of potential risks to the user. Type should be `ADEWARN`.
    - Outputs:
        - `MOTION_MODEL`: The modified AnimateDiff model with I2V features injected. This output represents the enhanced model ready for use in motion-based animations. Type should be `MOTION_MODEL_ADE`.
