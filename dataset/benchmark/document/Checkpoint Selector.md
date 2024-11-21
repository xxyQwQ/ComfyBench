- `Checkpoint Selector`: The Checkpoint Selector node is designed to facilitate the selection of checkpoint files from a predefined list. It abstracts the process of identifying and choosing the appropriate checkpoint file for operations such as model loading or initialization, streamlining the workflow for tasks that require specific checkpoint configurations.
    - Inputs:
        - `ckpt_name` (Required): Specifies the checkpoint file name to be selected. This parameter is crucial for determining which checkpoint file is to be used for subsequent operations, effectively guiding the node's output. Type should be `COMBO[STRING]`.
    - Outputs:
        - `ckpt_name`: Returns the selected checkpoint file name. This output is essential for downstream tasks that require a specific checkpoint file, such as model loading or further processing. Type should be `COMBO[STRING]`.