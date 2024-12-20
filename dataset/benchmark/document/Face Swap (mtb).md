- `Face Swap (mtb)`: Performs face swapping between a source and a target image using deep learning models, specifically designed to handle complex scenarios involving multiple faces and preserving facial features accurately.
    - Inputs:
        - `image` (Required): The image tensor where the face(s) will be swapped onto. It serves as the canvas for the operation, determining the final output's visual context. Type should be `IMAGE`.
        - `reference` (Required): The reference image tensor providing the face(s) to be swapped into the target image. It acts as the source of facial features for the swap. Type should be `IMAGE`.
        - `faces_index` (Required): A string specifying the indices of faces in the target image to be swapped. It allows selective swapping, enhancing control over the output. Type should be `STRING`.
        - `faceanalysis_model` (Required): The model used for analyzing faces within images, crucial for identifying and extracting facial features accurately. Type should be `FACE_ANALYSIS_MODEL`.
        - `faceswap_model` (Required): The model responsible for the actual face swapping process, leveraging deep learning to ensure realistic and seamless swaps. Type should be `FACESWAP_MODEL`.
        - `preserve_alpha` (Optional): A boolean indicating whether to preserve the alpha channel of the image, allowing for transparency handling in images with RGBA format. Type should be `BOOLEAN`.
    - Outputs:
        - `image`: The resulting image after the face swap has been performed, showcasing the swapped faces within the original image's context. Type should be `IMAGE`.
