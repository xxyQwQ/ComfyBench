# UNetSelfAttentionMultiply
## Documentation
- Class name: `UNetSelfAttentionMultiply`
- Category: `_for_testing/attention_experiments`
- Output node: `False`

This node specializes in modifying the self-attention mechanism within a U-Net model by applying custom scaling factors to the query, key, value, and output components of the attention mechanism. It aims to experimentally adjust the attention dynamics to explore different model behaviors or improve performance.
## Input types
### Required
- **`model`**
    - The U-Net model to be modified. It serves as the foundation for applying attention modifications, influencing the overall execution and results of the node.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`q`**
    - The scaling factor for the query component of the attention mechanism. It adjusts the influence of the query in the attention calculation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`k`**
    - The scaling factor for the key component of the attention mechanism. It modifies the impact of the key in determining the attention weights.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`v`**
    - The scaling factor for the value component of the attention mechanism. It affects how much each value contributes to the final output based on the attention weights.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`out`**
    - The scaling factor for the output of the attention mechanism. It influences the final output by scaling the aggregated values post-attention calculation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified U-Net model with adjusted self-attention mechanism. It reflects the changes made to the attention components through the specified scaling factors.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class UNetSelfAttentionMultiply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "q": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "k": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "v": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing/attention_experiments"

    def patch(self, model, q, k, v, out):
        m = attention_multiply("attn1", model, q, k, v, out)
        return (m, )

```
