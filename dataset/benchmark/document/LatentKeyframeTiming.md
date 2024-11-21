- `LatentKeyframeTiming`: This node is designed to manage the timing and sequencing of latent keyframes within a generative model's control network. It focuses on interpolating and scheduling keyframes based on specified timing parameters, ensuring smooth transitions and accurate timing for the generation process.
    - Inputs:
        - `batch_index_from` (Required): Specifies the starting index for the batch of keyframes to be processed, serving as the initial point for timing and sequencing operations. Type should be `INT`.
        - `batch_index_to_excl` (Required): Defines the exclusive end index for the batch of keyframes, marking the boundary for the sequence of keyframes to be interpolated or scheduled. Type should be `INT`.
        - `strength_from` (Required): Indicates the starting strength value for the interpolation of keyframes, setting the initial intensity or effect level. Type should be `FLOAT`.
        - `strength_to` (Required): Specifies the ending strength value for the interpolation, determining the final intensity or effect level for the sequence of keyframes. Type should be `FLOAT`.
        - `interpolation` (Required): Defines the method of interpolation to be used for transitioning between keyframes, such as linear or ease-in/out, to ensure smooth changes in strength or effects. Type should be `COMBO[STRING]`.
        - `prev_latent_kf` (Optional): Optional parameter for providing a previous set of latent keyframes to be considered or integrated into the current timing and sequencing operation. Type should be `LATENT_KEYFRAME`.
        - `print_keyframes` (Optional): A flag to enable logging of keyframe details for debugging or informational purposes. Type should be `BOOLEAN`.
        - `autosize` (Optional): unknown Type should be `ACNAUTOSIZE`.
    - Outputs:
        - `LATENT_KF`: The output is a modified or newly created set of latent keyframes, reflecting the applied timing and sequencing operations. Type should be `LATENT_KEYFRAME`.