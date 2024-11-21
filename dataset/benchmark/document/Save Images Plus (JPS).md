- `Save Images Plus (JPS)`: This node specializes in efficiently storing images on disk, offering capabilities for metadata customization and compression preferences. It facilitates organized file management and supports dynamic adaptation to image attributes, enhancing the storage process with optional metadata embedding for enriched file context.
    - Inputs:
        - `images` (Required): A batch of images to be saved. This parameter is crucial for determining the output file names and paths, as well as for the actual image saving process. Type should be `IMAGE`.
        - `filename_prefix` (Required): An optional prefix for the output filenames, allowing for organized storage and easy identification of saved images. Type should be `STRING`.
    - Outputs:
        - `dummy_out`: unknown Type should be `INT`.