- `Inference_Core_SavePoseKpsAsJsonFile`: This node is designed to save pose keypoints data as a JSON file, incorporating a filename prefix customization feature. It facilitates the storage of pose keypoints information, enabling further analysis or visualization of pose data.
    - Parameters:
        - `filename_prefix`: An optional prefix for the filename, allowing for easier identification and organization of saved files. Type should be `STRING`.
    - Inputs:
        - `pose_kps`: The pose keypoints to be saved. This data is crucial for capturing the spatial positions of various body parts in an image. Type should be `POSE_KEYPOINT`.
    - Outputs: