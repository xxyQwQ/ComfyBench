- `ADE_LoadAnimateLCMI2VModel`: The ADE_LoadAnimateLCMI2VModel node is designed to load and prepare AnimateLCM-I2V motion models for animation processes. It ensures that the selected motion model is compatible with the AnimateLCM-I2V format and has an image encoder, facilitating the integration of motion models into the animation workflow.
    - Inputs:
        - `model_name` (Required): Specifies the name of the motion model to be loaded, ensuring that the model is available and compatible with the AnimateLCM-I2V format. Type should be `COMBO[STRING]`.
        - `ad_settings` (Optional): Optional settings for the Animate Diff process that can be applied to the motion model, allowing for customization of the animation. Type should be `AD_SETTINGS`.
    - Outputs:
        - `MOTION_MODEL`: The loaded motion model, ready for use in animation processes. Type should be `MOTION_MODEL_ADE`.
        - `encoder_only`: A version of the motion model that only includes the encoder, useful for specific animation tasks. Type should be `MOTION_MODEL_ADE`.
