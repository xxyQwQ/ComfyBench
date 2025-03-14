- `UltralyticsDetectorProvider`: This node is designed to load and provide access to detection models, facilitating object detection tasks by leveraging models trained with the Ultralytics framework.
    - Inputs:
        - `model_name` (Required): Specifies the name of the model to be loaded, which is crucial for identifying and accessing the correct model file for object detection tasks. Type should be `COMBO[STRING]`.
    - Outputs:
        - `bbox_detector`: Provides an object detector that identifies bounding boxes around detected objects in images. Type should be `BBOX_DETECTOR`.
        - `segm_detector`: Offers a segmentation model capable of delineating the precise shape of objects by classifying each pixel of the image. Type should be `SEGM_DETECTOR`.
