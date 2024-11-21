- `SaveImageExtended`: The SaveImageExtended node extends the functionality of image saving in ComfyUI by allowing for enhanced customization and metadata handling. It supports saving images with custom filenames and paths, embedding metadata within the image files, and optionally saving job-related data alongside the images. This node caters to advanced use cases where detailed control over the image output process is desired, including the management of image metadata and the organization of saved images.
    - Inputs:
        - `images` (Required): A collection of images to be saved. This parameter is crucial as it directly influences the node's primary function of saving images to disk. Type should be `IMAGE`.
        - `filename_prefix` (Required): A prefix added to the filename for further customization, allowing for more descriptive or organized file naming. Type should be `STRING`.
        - `filename_keys` (Required): Specifies the keys from the generation parameters to be included in the filename, enabling dynamic naming based on specific generation attributes. Type should be `STRING`.
        - `foldername_prefix` (Required): A prefix for the folder name where images will be saved, allowing for organized grouping of images in the output directory. Type should be `STRING`.
        - `foldername_keys` (Required): Defines the keys from the generation parameters to be included in the folder name, facilitating organized storage based on specific attributes. Type should be `STRING`.
        - `delimiter` (Required): The character used to separate elements in the filename and folder name, allowing for customization of the file and folder naming scheme. Type should be `COMBO[STRING]`.
        - `save_job_data` (Required): Controls the saving of job-related data alongside the images, enabling the association of creation parameters or results with the saved images. Type should be `COMBO[STRING]`.
        - `job_data_per_image` (Required): Determines whether job-related data should be saved for each image individually or as a single file for all images, affecting how job data is organized. Type should be `COMBO[STRING]`.
        - `job_custom_text` (Required): Custom text to be included in job-related data, offering a way to embed arbitrary information or notes alongside the saved images. Type should be `STRING`.
        - `save_metadata` (Required): Controls whether to embed metadata within the saved images. This affects the node's ability to include additional information like prompts or custom data within the image files. Type should be `COMBO[STRING]`.
        - `counter_digits` (Required): Determines the number of digits for the image counter, affecting the formatting of the sequence numbers in filenames. Type should be `COMBO[INT]`.
        - `counter_position` (Required): Specifies the position of the counter in the filename, affecting how the sequence numbers are formatted and displayed. Type should be `COMBO[STRING]`.
        - `one_counter_per_folder` (Required): Specifies whether a single counter is used for all images in a folder or if each folder has its own counter, influencing file organization. Type should be `COMBO[STRING]`.
        - `image_preview` (Required): Controls whether a preview of the saved image is displayed, enhancing the user's ability to visually verify the saved images. Type should be `COMBO[STRING]`.
        - `positive_text_opt` (Optional): Optional text associated with positive prompts, used in job data saving to provide context or details about the image generation. Type should be `STRING`.
        - `negative_text_opt` (Optional): Optional text associated with negative prompts, used in job data saving to provide additional context or details about the image generation. Type should be `STRING`.
    - Outputs: