- `easy ultralyticsDetectorPipe`: This node is designed to integrate Ultralytics object detection models into a pipeline, facilitating the detection of bounding boxes and segmentation masks for objects within images. It leverages Ultralytics' YOLO models to analyze image content and identify objects, providing detailed spatial information that can be used for further image processing or analysis.
    - Inputs:
        - `model_name` (Required): Specifies the model to be used for detection, allowing for dynamic selection based on available Ultralytics YOLO models. This flexibility supports various detection needs by accommodating different model strengths and capabilities. Type should be `COMBO[STRING]`.
        - `bbox_threshold` (Required): Determines the confidence threshold for bounding box detections, filtering out detections with confidence scores below this value to ensure result relevance and accuracy. Type should be `FLOAT`.
        - `bbox_dilation` (Required): Adjusts the size of the detected bounding boxes by dilating or contracting them, enabling fine-tuning of object boundaries for subsequent processing steps. Type should be `INT`.
        - `bbox_crop_factor` (Required): Controls the extent to which the bounding box is enlarged or shrunk, providing a mechanism to include more or less context around detected objects. Type should be `FLOAT`.
    - Outputs:
        - `bbox_segm_pipe`: Outputs a pipeline configuration tailored for object detection, including model parameters and detection settings, ready for integration into broader image processing workflows. Type should be `PIPE_LINE`.