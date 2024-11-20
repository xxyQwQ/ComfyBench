# ModelPass
## Documentation
- Class name: `ModelPassThrough`
- Category: `KJNodes/misc`
- Output node: `False`

The ModelPassThrough node is designed to seamlessly forward a model input without any modifications. This functionality serves as a workaround for scenarios where bypassing inputs directly is not supported, ensuring that the model data can flow through the node unchanged.
## Input types
### Required
- **`model`**
    - The model input represents the model data that will be passed through unchanged. It is crucial for maintaining the integrity of the model's structure and data throughout the node's operation.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The output model is identical to the input model, having been passed through the node without any alterations. This ensures the model's structure and data remain intact for subsequent operations.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ModelPassThrough:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
            }, 
    }

    RETURN_TYPES = ("MODEL", )
    RETURN_NAMES = ("model",)
    FUNCTION = "passthrough"
    CATEGORY = "KJNodes/misc"
    DESCRIPTION = """
    Simply passes through the model,
    workaround for Set node not allowing bypassed inputs.
"""

    def passthrough(self, model):
        return (model,)

```
