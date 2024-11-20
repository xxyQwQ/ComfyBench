# ModelSamplingSD3
## Documentation
- Class name: `ModelSamplingSD3`
- Category: `advanced/model`
- Output node: `False`

This node introduces an advanced model sampling technique by applying a discrete flow-based modification to the input model. It enhances the model's sampling capabilities by integrating a shift parameter, allowing for refined control over the sampling process.
## Input types
### Required
- **`model`**
    - The model to which the sampling technique will be applied. It serves as the foundation for the advanced sampling modifications.
    - Comfy dtype: `MODEL`
    - Python dtype: `comfy.model_sampling.Model`
- **`shift`**
    - A parameter that adjusts the intensity of the sampling modification, offering a means to fine-tune the model's sampling behavior.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model with enhanced sampling capabilities, incorporating the discrete flow-based adjustments.
    - Python dtype: `comfy.model_sampling.Model`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ModelSamplingSD3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step":0.01}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, shift):
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)
        return (m, )

```
