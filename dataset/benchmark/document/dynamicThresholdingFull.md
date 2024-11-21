- `dynamicThresholdingFull`: The `dynamicThresholdingFull` node dynamically adjusts thresholding parameters for image processing tasks, leveraging inputs such as model, mimic scale, and threshold percentile. It employs dynamic thresholding to adaptively modify processing behavior, optimizing the balance between detail preservation and noise reduction.
    - Inputs:
        - `model` (Required): Specifies the model for dynamic thresholding, serving as the core component for threshold adjustments. Type should be `MODEL`.
        - `mimic_scale` (Required): Determines the scale at which the model mimics aspects of the input, influencing thresholding behavior. Type should be `FLOAT`.
        - `threshold_percentile` (Required): Sets the percentile for threshold calculation, affecting the aggressiveness of thresholding. Type should be `FLOAT`.
        - `mimic_mode` (Required): Defines the mode of mimicry, guiding how the mimic scale is applied. Type should be `COMBO[STRING]`.
        - `mimic_scale_min` (Required): Establishes the minimum scale for mimicry, ensuring a baseline level of detail preservation. Type should be `FLOAT`.
        - `cfg_mode` (Required): Specifies the configuration mode, altering thresholding behavior based on the model's configuration. Type should be `COMBO[STRING]`.
        - `cfg_scale_min` (Required): Indicates the minimum scale for configuration, impacting the fineness of thresholding adjustments. Type should be `FLOAT`.
        - `sched_val` (Required): A value to schedule or adjust the thresholding dynamically over time or iterations. Type should be `FLOAT`.
        - `separate_feature_channels` (Required): Determines whether feature channels should be processed separately or together, affecting the thresholding process. Type should be `COMBO[STRING]`.
        - `scaling_startpoint` (Required): Defines the starting point for scaling in the dynamic thresholding process. Type should be `COMBO[STRING]`.
        - `variability_measure` (Required): Specifies the measure of variability to consider in the dynamic thresholding process. Type should be `COMBO[STRING]`.
        - `interpolate_phi` (Required): A factor for interpolating the thresholding function, influencing the smoothness of the transition between thresholds. Type should be `FLOAT`.
    - Outputs:
        - `model`: Produces a modified model with dynamically adjusted thresholding parameters, optimized for image processing. Type should be `MODEL`.