- `ADE_CameraCtrlAnimateDiffKeyframe`: This node is designed to generate and manage keyframes for animations that incorporate camera control, enabling the creation of dynamic and complex camera movements within the AnimateDiff framework. It allows for the specification of start percentages for animations, the application of multiple values for scaling, effects, and camera control, and the inheritance of missing values to ensure continuity across keyframes.
    - Inputs:
        - `start_percent` (Required): Specifies the starting point of the animation as a percentage, allowing for precise control over the timing of camera movements and effects within the animation. Type should be `FLOAT`.
        - `prev_ad_keyframes` (Optional): Optional. Allows for the inclusion of previously defined AnimateDiff keyframes, enabling the chaining and layering of animations for more complex sequences. Type should be `AD_KEYFRAMES`.
        - `scale_multival` (Optional): Optional. Applies a scaling factor to the animation, enabling adjustments to the size of animated elements. Type should be `MULTIVAL`.
        - `effect_multival` (Optional): Optional. Applies various effects to the animation, enabling the addition of visual enhancements or modifications. Type should be `MULTIVAL`.
        - `cameractrl_multival` (Optional): Optional. Specifies multiple values for camera control, allowing for the creation of intricate camera movements within the animation. Type should be `MULTIVAL`.
        - `inherit_missing` (Optional): Determines whether missing values in the current keyframe should be inherited from previous keyframes, ensuring continuity in the animation. Type should be `BOOLEAN`.
        - `guarantee_steps` (Optional): Specifies the minimum number of steps to be guaranteed in the animation, ensuring a certain level of smoothness and continuity. Type should be `INT`.
    - Outputs:
        - `ad_keyframes`: Produces a sequence of AnimateDiff keyframes, enabling the creation of animations with complex camera movements. Type should be `AD_KEYFRAMES`.
