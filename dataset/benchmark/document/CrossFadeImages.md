- `CrossFadeImages`: The CrossFadeImages node is designed to blend two sequences of images together over a specified number of frames, using a variety of easing functions to control the transition dynamics. This node allows for the creation of smooth transitions between images, making it ideal for generating animations or video effects where a gradual change from one image set to another is desired.
    - Inputs:
        - `images_i` (Required): unknown Type should be `IMAGE`.
        - `interpolation` (Required): Specifies the easing function to be used for the transition, allowing for various types of dynamic effects such as linear, ease-in, ease-out, and more complex functions like bounce or elastic. Type should be `COMBO[STRING]`.
        - `transition_start_index` (Required): The index in the image sequences where the transition begins, allowing for precise control over the timing of the crossfade effect. Type should be `INT`.
        - `transitioning_frames` (Required): The number of frames over which the transition occurs, defining the length of the crossfade effect. Type should be `INT`.
        - `start_level` (Required): The initial alpha value for blending the images at the start of the transition, providing control over the beginning transparency level. Type should be `FLOAT`.
        - `end_level` (Required): The final alpha value for blending the images at the end of the transition, allowing for adjustment of the ending transparency level. Type should be `FLOAT`.
    - Outputs:
        - `image`: The resulting sequence of images after applying the crossfade transition, combining elements of both input sequences into a single smooth animation. Type should be `IMAGE`.