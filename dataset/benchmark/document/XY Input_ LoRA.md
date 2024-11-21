- `XY Input_ LoRA`: This node is designed to process and manipulate LoRA (Low-Rank Adaptation) parameters for generating XY plot inputs. It handles various LoRA-related inputs, such as batch information, model and clip strengths, and weights, to produce structured data suitable for visualization or further analysis. The node dynamically adjusts to different LoRA configurations, ensuring compatibility and flexibility for diverse analytical needs.
    - Inputs:
        - `input_mode` (Required): Specifies the operational mode of the node, affecting how LoRA parameters are processed and interpreted for XY plotting. Type should be `COMBO[STRING]`.
        - `batch_path` (Required): Defines the file path for batch processing of LoRA parameters, enabling the node to handle multiple inputs simultaneously for XY plotting. Type should be `STRING`.
        - `subdirectories` (Required): Indicates whether subdirectories should be considered in the batch processing of LoRA parameters, affecting the scope of data inclusion for XY plotting. Type should be `BOOLEAN`.
        - `batch_sort` (Required): Determines the sorting order of batch-processed LoRA parameters, influencing the organization of data for XY plotting. Type should be `COMBO[STRING]`.
        - `batch_max` (Required): Sets the maximum number of batches to be processed, limiting the volume of LoRA parameters considered for XY plotting. Type should be `INT`.
        - `lora_count` (Required): Specifies the number of LoRA parameters to be processed, directly impacting the composition of data for XY plotting. Type should be `INT`.
        - `model_strength` (Required): Defines the strength of the model adjustment for LoRA parameters, influencing the intensity of modifications applied for XY plotting. Type should be `FLOAT`.
        - `clip_strength` (Required): Sets the clip strength for LoRA parameters, affecting the extent of clipping applied during the processing for XY plotting. Type should be `FLOAT`.
        - `lora_name_i` (Required): Identifies individual LoRA parameters by name, allowing for specific selection and manipulation within the XY plotting process. Type should be `COMBO[STRING]`.
        - `model_str_i` (Required): Specifies the model strength for individual LoRA parameters, enabling fine-tuned control over the adjustment intensity for each. Type should be `FLOAT`.
        - `clip_str_i` (Required): Determines the clip strength for individual LoRA parameters, allowing for precise clipping adjustments on a per-parameter basis. Type should be `FLOAT`.
        - `lora_stack` (Optional): Represents an optional stack of LoRA parameters, contributing to the aggregated LoRA data for XY plotting. Type should be `LORA_STACK`.
    - Outputs:
        - `X or Y`: Outputs structured LoRA data for XY plotting, encapsulating various LoRA configurations and parameters in either the X or Y axis. Type should be `XY`.