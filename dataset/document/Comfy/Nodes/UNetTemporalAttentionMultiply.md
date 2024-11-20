# UNetTemporalAttentionMultiply
## Documentation
- Class name: `UNetTemporalAttentionMultiply`
- Category: `_for_testing/attention_experiments`
- Output node: `False`

This node specializes in applying temporal attention modifications to U-Net models, enhancing their ability to focus on relevant temporal features within the data. It leverages attention mechanisms to dynamically adjust the model's focus, aiming to improve performance on tasks that benefit from understanding temporal relationships.
## Input types
### Required
- **`model`**
    - The U-Net model to which temporal attention modifications will be applied. This parameter is crucial as it determines the base model that will undergo the attention-based transformation, directly influencing the outcome.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`self_structural`**
    - Adjusts the model's attention towards structural aspects within the same temporal context, enhancing its ability to discern structural features.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`self_temporal`**
    - Modifies the model's attention to focus more on temporal features within the same structural context, aiming to improve temporal feature recognition.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cross_structural`**
    - Enhances the model's ability to identify structural features across different temporal contexts, improving cross-temporal structural analysis.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cross_temporal`**
    - Aims to improve the model's focus on temporal features across different structural contexts, enhancing its temporal analysis capabilities.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified U-Net model with enhanced temporal attention capabilities, aimed at improving its performance on tasks requiring an understanding of temporal relationships.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class UNetTemporalAttentionMultiply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "self_structural": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "self_temporal": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "cross_structural": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "cross_temporal": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing/attention_experiments"

    def patch(self, model, self_structural, self_temporal, cross_structural, cross_temporal):
        m = model.clone()
        sd = model.model_state_dict()

        for k in sd:
            if (k.endswith("attn1.to_out.0.bias") or k.endswith("attn1.to_out.0.weight")):
                if '.time_stack.' in k:
                    m.add_patches({k: (None,)}, 0.0, self_temporal)
                else:
                    m.add_patches({k: (None,)}, 0.0, self_structural)
            elif (k.endswith("attn2.to_out.0.bias") or k.endswith("attn2.to_out.0.weight")):
                if '.time_stack.' in k:
                    m.add_patches({k: (None,)}, 0.0, cross_temporal)
                else:
                    m.add_patches({k: (None,)}, 0.0, cross_structural)
        return (m, )

```
