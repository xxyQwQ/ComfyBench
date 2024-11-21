- `APersonFaceLandmarkMaskGenerator`: The APersonFaceLandmarkMaskGenerator node is designed to generate facial landmark masks for images. It utilizes the MediaPipe Face Mesh solution to detect facial landmarks and then creates masks for specific facial features such as the face oval, lips, eyes, and eyebrows. This node is capable of processing multiple faces within an image, allowing for the generation of masks for each detected face.
    - Inputs:
        - `images` (Required): The input images for which facial landmark masks are to be generated. This parameter is essential for the detection and mask generation process, as the images undergo preprocessing and are fed into the MediaPipe Face Mesh model for landmark detection. Type should be `IMAGE`.
        - `face` (Optional): A boolean flag indicating whether to generate a mask for the face oval. This option allows for selective mask generation based on the user's requirements. Type should be `BOOLEAN`.
        - `left_eyebrow` (Optional): A boolean flag indicating whether to generate a mask for the left eyebrow. This option contributes to the comprehensive facial feature masking by including the left eyebrow. Type should be `BOOLEAN`.
        - `right_eyebrow` (Optional): A boolean flag indicating whether to generate a mask for the right eyebrow. Including this option ensures the right eyebrow is also considered in the mask generation process. Type should be `BOOLEAN`.
        - `left_eye` (Optional): A boolean flag indicating whether to generate a mask for the left eye. This enhances the mask generation by including detailed masks for the left eye. Type should be `BOOLEAN`.
        - `right_eye` (Optional): A boolean flag indicating whether to generate a mask for the right eye. This option adds to the facial feature coverage by generating masks for the right eye. Type should be `BOOLEAN`.
        - `left_pupil` (Optional): A boolean flag indicating whether to generate a mask for the left pupil. Activating this option allows for the inclusion of the left pupil in the facial feature masks. Type should be `BOOLEAN`.
        - `right_pupil` (Optional): A boolean flag indicating whether to generate a mask for the right pupil. This ensures the right pupil is also covered in the generated facial feature masks. Type should be `BOOLEAN`.
        - `lips` (Optional): A boolean flag indicating whether to generate a mask for the lips. Enabling this option results in the creation of detailed masks for the lips, enhancing the overall mask generation. Type should be `BOOLEAN`.
        - `number_of_faces` (Optional): Specifies the maximum number of faces to detect in the input images. This parameter influences the scope of face detection, thereby affecting the number and detail of the generated masks. Type should be `INT`.
        - `confidence` (Optional): The minimum confidence threshold for detecting faces. A higher value means that only faces with a higher likelihood of being correctly identified will be processed, impacting the accuracy and completeness of the generated masks. Type should be `FLOAT`.
    - Outputs:
        - `masks`: The output is a tensor containing the generated masks for the specified facial features. Each mask corresponds to a different facial feature or face detected in the input images. Type should be `MASK`.