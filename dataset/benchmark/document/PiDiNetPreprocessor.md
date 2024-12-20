- `PiDiNetPreprocessor`: The PiDiNetPreprocessor node is designed for preprocessing images to extract soft-edge lines, utilizing the PiDiNet model. It aims to enhance image analysis and processing tasks by providing a refined input for further processing or analysis.
    - Inputs:
        - `image` (Required): The input image to be processed for soft-edge line extraction. It's the primary data the node operates on, aiming to enhance its features for subsequent analysis. Type should be `IMAGE`.
        - `safe` (Optional): A mode selector that determines whether to perform the operation in a safe mode. Enabling 'safe' mode ensures that the preprocessing adheres to safety constraints, potentially affecting the output's fidelity and detail in favor of reducing risks during processing. Type should be `COMBO[STRING]`.
        - `resolution` (Optional): Specifies the resolution at which the image should be processed. A higher resolution leads to more detailed extraction of soft-edge lines, while a lower resolution might speed up the process at the cost of detail. Type should be `INT`.
    - Outputs:
        - `image`: The processed image with extracted soft-edge lines, ready for further analysis or processing steps. Type should be `IMAGE`.
