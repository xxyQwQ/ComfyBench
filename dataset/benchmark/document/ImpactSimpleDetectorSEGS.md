- `ImpactSimpleDetectorSEGS`: The ImpactSimpleDetectorSEGS node is designed to perform segmentation detection on images, identifying and isolating specific segments within an image. This node applies advanced segmentation models to analyze images, detect distinct segments based on visual content, and output detailed segmentation information, facilitating further image analysis or processing tasks.
    - Inputs:
        - `bbox_detector` (Required): Specifies the bounding box detector used for initial detection before segmentation. It's crucial for identifying areas of interest within the image that will be further processed for segmentation. Type should be `BBOX_DETECTOR`.
        - `image` (Required): unknown Type should be `IMAGE`.
        - `bbox_threshold` (Required): A threshold value for the bounding box detection phase, determining the sensitivity of detecting potential areas of interest. Type should be `FLOAT`.
        - `bbox_dilation` (Required): Specifies the dilation level applied to the bounding boxes detected, affecting the area considered for segmentation. Type should be `INT`.
        - `crop_factor` (Required): Determines how much the detected segments are cropped around the bounding boxes, influencing the focus area for segmentation. Type should be `FLOAT`.
        - `drop_size` (Required): Defines the minimum size for segments to be considered valid, filtering out smaller, potentially irrelevant segments. Type should be `INT`.
        - `sub_threshold` (Required): A secondary threshold used for finer segmentation within the initially detected areas, refining the segmentation process. Type should be `FLOAT`.
        - `sub_dilation` (Required): The dilation level applied during the secondary segmentation phase, further adjusting the clarity of segment boundaries. Type should be `INT`.
        - `sub_bbox_expansion` (Required): Specifies how much the bounding boxes are expanded during the secondary segmentation phase, allowing for more comprehensive segment detection. Type should be `INT`.
        - `sam_mask_hint_threshold` (Required): A threshold value used in conjunction with SAM models to refine segmentation based on semantic cues. Type should be `FLOAT`.
        - `post_dilation` (Optional): unknown Type should be `INT`.
        - `sam_model_opt` (Optional): unknown Type should be `SAM_MODEL`.
        - `segm_detector_opt` (Optional): unknown Type should be `SEGM_DETECTOR`.
    - Outputs:
        - `segs`: The output segmentation details, including the shapes and characteristics of the detected segments, enabling further analysis or modification. Type should be `SEGS`.
