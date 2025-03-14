- `XY Input_ Checkpoint`: The TSC_XYplot_Checkpoint node is designed to process XY plot data related to model checkpoints, focusing on validating and handling checkpoint values and clip skip values for efficiency analysis in model training and refinement processes.
    - Inputs:
        - `target_ckpt` (Required): Specifies the target checkpoint type, either 'Base' or 'Refiner', to determine the context of the XY plot data processing. Type should be `COMBO[STRING]`.
        - `input_mode` (Required): Determines the mode of input which affects how checkpoint data is processed and visualized in the XY plot. Type should be `COMBO[STRING]`.
        - `batch_path` (Required): The file path to the batch data used for generating the XY plot, influencing the source of data analysis. Type should be `STRING`.
        - `subdirectories` (Required): A boolean indicating whether to include subdirectories in the batch data search, expanding the scope of data analysis. Type should be `BOOLEAN`.
        - `batch_sort` (Required): Specifies the sorting order of batch data, either 'ascending' or 'descending', to organize the data for analysis. Type should be `COMBO[STRING]`.
        - `batch_max` (Required): Defines the maximum number of batches to consider for the XY plot, setting an upper limit on the data analysis. Type should be `INT`.
        - `ckpt_count` (Required): Indicates the number of checkpoints to include in the analysis, directly impacting the comprehensiveness of the XY plot. Type should be `INT`.
        - `ckpt_name_i` (Required): Specifies the name of the i-th checkpoint, allowing for detailed customization of checkpoint data included in the XY plot. Type should be `COMBO[STRING]`.
        - `clip_skip_i` (Required): Determines the clip skip value for the i-th checkpoint, affecting the granularity of data analysis for each checkpoint. Type should be `INT`.
        - `vae_name_i` (Required): Identifies the name of the i-th VAE model, enabling the inclusion of VAE-specific data in the XY plot analysis. Type should be `COMBO[STRING]`.
    - Outputs:
        - `X or Y`: Outputs the type of XY plot data generated ('Clip Skip' or 'Clip Skip (Refiner)') along with the corresponding values, facilitating efficiency analysis. Type should be `XY`.
