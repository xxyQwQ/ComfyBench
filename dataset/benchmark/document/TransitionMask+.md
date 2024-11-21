- `TransitionMask+`: The TransitionMask+ node specializes in generating dynamic transition effects within masks, offering a range of transition styles and timing functions to create complex visual transitions between frames or states. This node simplifies the creation of animated or static transition effects, making it easier for users to apply sophisticated mask transitions in their projects.
    - Inputs:
        - `width` (Required): Defines the width of the mask to be generated, setting the horizontal dimension of the transition effect. Type should be `INT`.
        - `height` (Required): Sets the height of the mask, determining the vertical dimension of the transition effect. Type should be `INT`.
        - `frames` (Required): Specifies the total number of frames in the transition animation, controlling the length of the transition effect. Type should be `INT`.
        - `start_frame` (Required): Indicates the starting frame number for the transition effect, allowing for control over the animation's beginning. Type should be `INT`.
        - `end_frame` (Required): Determines the ending frame number for the transition, enabling customization of the animation's duration. Type should be `INT`.
        - `transition_type` (Required): Selects the type of transition effect to be applied, such as slides, bars, boxes, or fades, offering a variety of visual styles. Type should be `COMBO[STRING]`.
        - `timing_function` (Required): Chooses the timing function for the transition effect, such as linear or ease-in-out, affecting the pacing of the transition. Type should be `COMBO[STRING]`.
    - Outputs:
        - `mask`: Produces a mask that represents the transition effect, which can be used to apply or visualize the transition within an image or a series of images. Type should be `MASK`.