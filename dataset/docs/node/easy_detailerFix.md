- `easy detailerFix`: The `easy detailerFix` node is designed to enhance and refine the details of generated images, focusing on improving visual quality and coherence. It applies a series of adjustments and fixes to the image, aiming to correct any imperfections and enhance overall detail, making it an essential step for achieving high-quality, polished outputs.
    - Parameters:
        - `image_output`: Determines the output format for the images, affecting how results are displayed or saved. Type should be `COMBO[STRING]`.
        - `link_id`: Identifies the link for sending images, used when the output is configured to be sent to a specific destination. Type should be `INT`.
        - `save_prefix`: Sets the prefix for saved images, organizing output files according to user-defined naming conventions. Type should be `STRING`.
    - Inputs:
        - `pipe`: Specifies the pipeline configuration to be used for the detailer fix process, which includes model and processing settings. Type should be `PIPE_LINE`.
        - `model`: Optionally specifies a model to be used for the detailer fix process, allowing for customization of enhancements. Type should be `MODEL`.
    - Outputs:
        - `pipe`: Outputs the updated pipeline configuration after applying detail fixes. Type should be `PIPE_LINE`.
        - `image`: The enhanced image after detail fixes have been applied. Type should be `IMAGE`.
        - `cropped_refined`: The refined cropped version of the enhanced image. Type should be `IMAGE`.
        - `cropped_enhanced_alpha`: The cropped image with enhanced details and alpha transparency. Type should be `IMAGE`.