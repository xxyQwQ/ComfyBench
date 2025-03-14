- `Stack Images`: The Stack Images node is designed to aggregate multiple images into a single composite image. It supports various stacking modes and configurations, including the ability to handle batches of images, apply specific stacking directions, and incorporate optional labels for both horizontal and vertical orientations. This functionality is particularly useful for visualizing collections of images in a structured and coherent manner.
    - Inputs:
        - `images` (Required): A list of images to be stacked together. This parameter is crucial for defining the set of images that will be processed and combined into a single composite image. Type should be `IMAGE`.
        - `splits` (Required): Defines how the images are split into batches or groups for stacking. This parameter influences the organization of images within the final composite image. Type should be `INT`.
        - `stack_mode` (Required): Specifies the direction (horizontal or vertical) in which the images within a batch are stacked. This affects the layout of the composite image. Type should be `COMBO[STRING]`.
        - `batch_stack_mode` (Required): Determines the stacking direction of batches of images. This parameter is essential for defining the overall structure of the composite image. Type should be `COMBO[STRING]`.
        - `horizontal_labels` (Optional): Optional labels for the horizontal axis, used when the stacking direction allows for horizontal labeling. Enhances the interpretability of the composite image. Type should be `*`.
        - `vertical_labels` (Optional): Optional labels for the vertical axis, used when the stacking direction allows for vertical labeling. Enhances the interpretability of the composite image. Type should be `*`.
    - Outputs:
        - `Image`: The resulting composite image after stacking. Type should be `IMAGE`.
