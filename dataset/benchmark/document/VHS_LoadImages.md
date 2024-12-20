- `VHS_LoadImages`: The node 'VHS_LoadImages' is designed for loading a batch of images from a directory, processing them according to specified parameters such as image load cap, skipping initial images, and selecting every nth image. It facilitates the handling of image data for further processing or analysis within a video helper suite, streamlining the workflow for video and image manipulation tasks.
    - Inputs:
        - `directory` (Required): Specifies the directory path from which images are to be loaded. It is crucial for determining the source of the images to be processed. Type should be `COMBO[STRING]`.
        - `image_load_cap` (Optional): Limits the number of images to be loaded from the directory, optimizing resource usage and processing time. Type should be `INT`.
        - `skip_first_images` (Optional): Skips a specified number of initial images in the directory, allowing for flexible data handling and selection. Type should be `INT`.
        - `select_every_nth` (Optional): Loads every nth image from the directory, providing a mechanism for sampling or reducing the dataset size. Type should be `INT`.
        - `meta_batch` (Optional): Optional parameter for integrating with a meta batch processing system, enabling batch processing of images. Type should be `VHS_BatchManager`.
    - Outputs:
        - `IMAGE`: The processed images loaded from the directory, ready for further manipulation or analysis. Type should be `IMAGE`.
        - `MASK`: Generated masks for the loaded images, applicable when images have an alpha channel for transparency. Type should be `MASK`.
        - `frame_count`: The total number of images successfully loaded and processed. Type should be `INT`.
