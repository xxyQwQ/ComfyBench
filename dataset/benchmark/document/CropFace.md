- `CropFace`: The CropFace node is designed for processing images by detecting faces, cropping them, and then applying a face restoration model to enhance or restore the cropped face images. It utilizes face detection to identify faces within an image, crops these faces, and then processes each cropped face through a face restoration model to improve image quality or restore facial details.
    - Inputs:
        - `image` (Required): The input image to be processed for face detection and restoration. Type should be `IMAGE`.
        - `facedetection` (Required): The face detection model used to identify and locate faces within the input image. Type should be `COMBO[STRING]`.
    - Outputs:
        - `image`: The output is a tensor of the cropped and restored faces, ready for further processing or visualization. Type should be `IMAGE`.
