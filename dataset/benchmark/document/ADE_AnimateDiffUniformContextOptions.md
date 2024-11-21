- `ADE_AnimateDiffUniformContextOptions`: This node is designed to provide uniform context options for the AnimateDiff process, facilitating the generation of animations with consistent context across frames. It is marked as legacy, indicating that it may have been superseded by newer methods or nodes for managing context in animation generation.
    - Inputs:
        - `context_length` (Required): Defines the length of the context used in the animation process, influencing the amount of information considered for generating each frame. Type should be `INT`.
        - `context_stride` (Required): Determines the stride between contexts in the animation, affecting the transition and flow between frames. Type should be `INT`.
        - `context_overlap` (Required): Specifies the degree of overlap between contexts of consecutive frames, affecting the smoothness and continuity of the animation. Type should be `INT`.
        - `context_schedule` (Required): Specifies the scheduling method for context application, influencing the timing and sequence of animation frames. Type should be `COMBO[STRING]`.
        - `closed_loop` (Required): Indicates whether the animation should loop back to the beginning after completing, creating a continuous loop effect. Type should be `BOOLEAN`.
        - `fuse_method` (Optional): Determines the method used to fuse multiple contexts together, impacting the overall coherence and flow of the generated animation. Type should be `COMBO[STRING]`.
        - `use_on_equal_length` (Optional): A boolean flag indicating whether the context options should be applied uniformly across frames of equal length, ensuring consistent animation characteristics. Type should be `BOOLEAN`.
        - `start_percent` (Optional): Indicates the starting point of the animation as a percentage, allowing for precise control over the animation's initiation within the context. Type should be `FLOAT`.
        - `guarantee_steps` (Optional): Guarantees a minimum number of steps in the animation, ensuring that the animation reaches a certain length regardless of other parameters. Type should be `INT`.
        - `prev_context` (Optional): The previous context options group, which can be used as a basis for generating the new context options, facilitating continuity across animations. Type should be `CONTEXT_OPTIONS`.
        - `view_opts` (Optional): View options that specify additional parameters for rendering the animation, further customizing the output. Type should be `VIEW_OPTS`.
        - `deprecation_warning` (Optional): A notice that this node is considered legacy and may be replaced by newer methods or nodes in future updates. Type should be `ADEWARN`.
    - Outputs:
        - `CONTEXT_OPTS`: The generated context options, ready to be utilized in the AnimateDiff process for creating uniform animations. Type should be `CONTEXT_OPTIONS`.