- `FaceAlign`: The FaceAlign node is designed for aligning faces within images based on facial keypoints. It adjusts the orientation of a face in an input image to match the orientation of a face in a target image or to a standard alignment if no target is provided, enhancing the consistency of facial analysis or recognition tasks.
    - Inputs:
        - `analysis_models` (Required): Specifies the models used for facial analysis, particularly for detecting facial keypoints. Its role is crucial in determining the orientation and alignment of faces within images. Type should be `ANALYSIS_MODELS`.
        - `image_from` (Required): The source image containing the face to be aligned. This image is processed to detect facial keypoints and adjust its orientation based on these points. Type should be `IMAGE`.
        - `image_to` (Optional): An optional target image used to align the source image's face orientation with the target's face orientation. If provided, the alignment is adjusted to match the target face's orientation. Type should be `IMAGE`.
    - Outputs:
        - `image`: The aligned image with the face orientation adjusted either to match the target image or to a standard alignment if no target is provided. Type should be `IMAGE`.