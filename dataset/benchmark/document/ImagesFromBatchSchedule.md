- `ImagesFromBatchSchedule`: This node is designed to generate a batch of images based on a predefined schedule. It leverages a batch value schedule list to select and export images from an input batch, facilitating the creation of image sequences or animations. The node's functionality is rooted in the ability to process and manipulate image batches according to specified scheduling criteria, making it a valuable tool for dynamic image generation and manipulation tasks.
    - Inputs:
        - `images` (Required): Specifies the input batch of images to be processed and scheduled for output. This parameter is crucial for defining the source images that will be manipulated and organized according to the batch schedule. Type should be `IMAGE`.
        - `text` (Required): Defines the scheduling criteria or instructions as a text input, used to determine how images are selected and organized in the output batch. Type should be `STRING`.
        - `current_frame` (Required): Indicates the current frame number in the sequence, used to track progression and manage image selection according to the schedule. Type should be `INT`.
        - `max_frames` (Required): Sets the maximum number of frames to be included in the output batch, limiting the sequence length according to this threshold. Type should be `INT`.
        - `print_output` (Required): A boolean flag that, when set to True, enables the printing of output information for debugging or informational purposes. Type should be `BOOLEAN`.
    - Outputs:
        - `image`: The output is a batch of images that have been selected and organized according to the specified schedule, ready for further processing or visualization. Type should be `IMAGE`.