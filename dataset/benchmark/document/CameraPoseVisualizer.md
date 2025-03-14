- `CameraPoseVisualizer`: The CameraPoseVisualizer node is designed to visualize camera poses in a 3D plot, either from a provided text file containing camera intrinsics and coordinates or directly from camera control poses. It supports customization of the visualization through parameters such as scale, base value adjustments, and the option to use exact focal lengths. This visualization aids in understanding and analyzing the spatial orientation and field of view of cameras in a given scene.
    - Inputs:
        - `pose_file_path` (Required): Specifies the path to a text file containing camera poses or can be left empty to use camera control poses directly. It is essential for determining the source of camera poses to visualize. Type should be `STRING`.
        - `base_xval` (Required): A base value for x-axis adjustments in the visualization, allowing for fine-tuning of the camera's position. Type should be `FLOAT`.
        - `zval` (Required): A base value for z-axis adjustments, influencing the depth positioning in the visualization. Type should be `FLOAT`.
        - `scale` (Required): Scales the entire visualization, affecting the size and spacing of visualized camera poses. Type should be `FLOAT`.
        - `use_exact_fx` (Required): Determines whether to use exact focal lengths from the camera poses or a default value, impacting the accuracy of the visualization. Type should be `BOOLEAN`.
        - `relative_c2w` (Required): Controls whether camera-to-world transformations are considered relative, affecting the positioning and orientation of cameras. Type should be `BOOLEAN`.
        - `use_viewer` (Required): Enables or disables the use of an interactive viewer for the visualization, enhancing user interaction. Type should be `BOOLEAN`.
        - `cameractrl_poses` (Optional): Directly provides camera control poses for visualization, offering an alternative to loading poses from a file. Type should be `CAMERACTRL_POSES`.
    - Outputs:
        - `image`: Generates a 3D plot image visualizing the camera poses, providing a visual representation of camera orientations and positions. Type should be `IMAGE`.
