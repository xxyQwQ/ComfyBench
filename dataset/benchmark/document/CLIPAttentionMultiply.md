- `CLIPAttentionMultiply`: This node specializes in adjusting the attention mechanism within a CLIP model by applying multiplicative modifications to the attention's query, key, value, and output projections. It enables fine-tuning of the attention weights to potentially enhance model performance or adapt the model to specific tasks.
    - Inputs:
        - `clip` (Required): The CLIP model to be modified. It serves as the base for applying attention patches. Type should be `CLIP`.
        - `q` (Required): The multiplicative factor for the query projection weights and biases, influencing how the model attends to different parts of the input. Type should be `FLOAT`.
        - `k` (Required): The multiplicative factor for the key projection weights and biases, affecting the model's ability to match queries to keys. Type should be `FLOAT`.
        - `v` (Required): The multiplicative factor for the value projection weights and biases, impacting the content that gets prioritized in the attention output. Type should be `FLOAT`.
        - `out` (Required): The multiplicative factor for the output projection weights and biases, determining the final output of the attention mechanism. Type should be `FLOAT`.
    - Outputs:
        - `clip`: The modified CLIP model with adjusted attention weights, ready for further use or evaluation. Type should be `CLIP`.
