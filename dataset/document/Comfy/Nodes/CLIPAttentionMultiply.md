# CLIPAttentionMultiply
## Documentation
- Class name: `CLIPAttentionMultiply`
- Category: `_for_testing/attention_experiments`
- Output node: `False`

This node specializes in adjusting the attention mechanism within a CLIP model by applying multiplicative modifications to the attention's query, key, value, and output projections. It enables fine-tuning of the attention weights to potentially enhance model performance or adapt the model to specific tasks.
## Input types
### Required
- **`clip`**
    - The CLIP model to be modified. It serves as the base for applying attention patches.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`q`**
    - The multiplicative factor for the query projection weights and biases, influencing how the model attends to different parts of the input.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`k`**
    - The multiplicative factor for the key projection weights and biases, affecting the model's ability to match queries to keys.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`v`**
    - The multiplicative factor for the value projection weights and biases, impacting the content that gets prioritized in the attention output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`out`**
    - The multiplicative factor for the output projection weights and biases, determining the final output of the attention mechanism.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`clip`**
    - Comfy dtype: `CLIP`
    - The modified CLIP model with adjusted attention weights, ready for further use or evaluation.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CLIPAttentionMultiply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip": ("CLIP",),
                              "q": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "k": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "v": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing/attention_experiments"

    def patch(self, clip, q, k, v, out):
        m = clip.clone()
        sd = m.patcher.model_state_dict()

        for key in sd:
            if key.endswith("self_attn.q_proj.weight") or key.endswith("self_attn.q_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, q)
            if key.endswith("self_attn.k_proj.weight") or key.endswith("self_attn.k_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, k)
            if key.endswith("self_attn.v_proj.weight") or key.endswith("self_attn.v_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, v)
            if key.endswith("self_attn.out_proj.weight") or key.endswith("self_attn.out_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, out)
        return (m, )

```
