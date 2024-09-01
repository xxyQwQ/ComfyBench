- `Inference_Core_PiDiNetPreprocessor`: The PiDiNet Preprocessor node is designed for preprocessing images to extract soft-edge lines, utilizing the PiDiNet model for enhanced line detection. It supports configurable safety modes and resolution settings to adapt to various image processing needs.
    - Parameters:
        - `safe`: A mode that enables or disables safety checks during image processing, affecting the execution path and potentially the output quality. Type should be `COMBO[STRING]`.
        - `resolution`: The resolution to which the input image is resized before processing, impacting the detail level of the extracted lines. Type should be `INT`.
    - Inputs:
        - `image`: The input image to be processed for line extraction. It is the primary data on which the PiDiNet model operates. Type should be `IMAGE`.
    - Outputs:
        - `image`: The processed image with extracted lines, showcasing the capabilities of the PiDiNet model in enhancing line detection. Type should be `IMAGE`.