- `easy cascadeKSampler`: The easy cascadeKSampler node facilitates the generation of images through a cascading sampling process, leveraging a pipeline of models for enhanced detail and quality. It simplifies the complex setup typically required for cascading sampling, making it accessible for users to produce high-quality images with minimal configuration.
    - Inputs:
        - `pipe` (Required): Specifies the pipeline configuration for the cascading sampling process, including model and processing steps. It's crucial for defining the sequence of operations and models used. Type should be `PIPE_LINE`.
        - `image_output` (Required): Determines how the output images are handled, offering options for hiding, previewing, saving, or sending the images. Type should be `COMBO[STRING]`.
        - `link_id` (Required): An identifier used for linking the output with external systems or files, especially when sending or saving images. Type should be `INT`.
        - `save_prefix` (Required): A prefix added to saved images, allowing for organized storage and easy retrieval. Type should be `STRING`.
        - `model_c` (Optional): An optional model configuration that can be provided to customize the sampling process. Type should be `MODEL`.
    - Outputs:
        - `pipe`: Returns the updated pipeline configuration after the cascading sampling process, reflecting any changes or additions. Type should be `PIPE_LINE`.
        - `image`: The generated image as a result of the cascading sampling process. Type should be `IMAGE`.