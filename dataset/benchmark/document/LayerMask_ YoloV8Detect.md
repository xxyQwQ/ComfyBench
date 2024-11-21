- `LayerMask_ YoloV8Detect`: The YoloV8Detect node is designed for object detection within images, utilizing the YOLOv8 model to identify and segment objects. It supports the generation of masks for detected objects, allowing for selective application or removal of effects based on the presence of specific objects within an image. This node can handle multiple images, merge detected object masks according to specified criteria, and return both the original images with detected objects highlighted and the corresponding masks.
    - Inputs:
        - `image` (Required): The input image on which object detection and segmentation are to be performed. It is crucial for identifying and segmenting objects within the image using the YOLOv8 model. Type should be `IMAGE`.
        - `yolo_model` (Required): Specifies the YOLOv8 model to be used for object detection. This parameter is essential for configuring the detection process according to the model's capabilities and requirements. Type should be `COMBO[STRING]`.
        - `mask_merge` (Required): Determines how detected object masks should be merged. It can either merge all masks into one or merge a specified number of masks, affecting the final mask output. Type should be `COMBO[STRING]`.
    - Outputs:
        - `mask`: The final merged mask for all processed images, suitable for further processing or visualization. Type should be `MASK`.
        - `yolo_plot_image`: Images with detected objects highlighted, useful for visualizing the detection results. Type should be `IMAGE`.
        - `yolo_masks`: Individual masks for detected objects, before any merging is applied. Useful for detailed analysis or selective processing. Type should be `MASK`.