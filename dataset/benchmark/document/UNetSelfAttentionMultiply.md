- `UNetSelfAttentionMultiply`: This node specializes in modifying the self-attention mechanism within a U-Net model by applying custom scaling factors to the query, key, value, and output components of the attention mechanism. It aims to experimentally adjust the attention dynamics to explore different model behaviors or improve performance.
    - Inputs:
        - `model` (Required): The U-Net model to be modified. It serves as the foundation for applying attention modifications, influencing the overall execution and results of the node. Type should be `MODEL`.
        - `q` (Required): The scaling factor for the query component of the attention mechanism. It adjusts the influence of the query in the attention calculation. Type should be `FLOAT`.
        - `k` (Required): The scaling factor for the key component of the attention mechanism. It modifies the impact of the key in determining the attention weights. Type should be `FLOAT`.
        - `v` (Required): The scaling factor for the value component of the attention mechanism. It affects how much each value contributes to the final output based on the attention weights. Type should be `FLOAT`.
        - `out` (Required): The scaling factor for the output of the attention mechanism. It influences the final output by scaling the aggregated values post-attention calculation. Type should be `FLOAT`.
    - Outputs:
        - `model`: The modified U-Net model with adjusted self-attention mechanism. It reflects the changes made to the attention components through the specified scaling factors. Type should be `MODEL`.
