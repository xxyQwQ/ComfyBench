- `AnimeLineArtPreprocessor`: The AnimeLineArtPreprocessor node is designed for extracting line art from anime images. It utilizes a specialized model to process images and enhance their line art features, making it suitable for applications that require clean and distinct line drawings.
    - Inputs:
        - `image` (Required): The input image to be processed for line art extraction. This parameter is crucial as it determines the quality and characteristics of the output line art. Type should be `IMAGE`.
        - `resolution` (Optional): Specifies the resolution of the output image. A higher resolution can lead to more detailed line art but may increase processing time. Type should be `INT`.
    - Outputs:
        - `image`: The processed image with enhanced line art features. This output is ideal for further artistic or analytical applications. Type should be `IMAGE`.
