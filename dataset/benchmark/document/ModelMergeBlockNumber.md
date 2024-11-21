- `ModelMergeBlockNumber`: This node specializes in merging two models by blending their components based on specified ratios for different parts of the models. It allows for fine-grained control over the merging process by enabling the adjustment of blend ratios for various blocks within the models.
    - Inputs:
        - `model1` (Required): The first model to be merged. It serves as the base model onto which patches from the second model are applied. Type should be `MODEL`.
        - `model2` (Required): The second model to be merged. Key patches from this model are applied to the first model based on specified ratios. Type should be `MODEL`.
        - `time_embed.` (Required): Specifies the blend ratio for the time embedding components of the models. Type should be `FLOAT`.
        - `label_emb.` (Required): Specifies the blend ratio for the label embedding components of the models. Type should be `FLOAT`.
        - `input_blocks.i.` (Required): Specifies the blend ratio for each of the input blocks. The index 'i' ranges from 0 to 11, allowing for individual adjustment of 12 input blocks. Type should be `FLOAT`.
        - `middle_block.i.` (Required): Specifies the blend ratio for each of the middle blocks. The index 'i' ranges from 0 to 2, targeting 3 middle blocks for adjustment. Type should be `FLOAT`.
        - `output_blocks.i.` (Required): Specifies the blend ratio for each of the output blocks. The index 'i' ranges from 0 to 11, enabling individual adjustment of 12 output blocks. Type should be `FLOAT`.
        - `out.` (Required): Specifies the blend ratio for the final output components of the models. Type should be `FLOAT`.
    - Outputs:
        - `model`: The result of merging two models based on the specified blend ratios for various components. Type should be `MODEL`.