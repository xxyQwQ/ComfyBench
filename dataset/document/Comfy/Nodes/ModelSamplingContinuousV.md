# ModelSamplingContinuousV
## Documentation
- Class name: `ModelSamplingContinuousV`
- Category: `advanced/model`
- Output node: `False`

This node is designed to enhance a given model by applying a continuous variable (V) sampling technique, which involves cloning the original model and integrating a specialized sampling mechanism based on the 'v_prediction' method. It adjusts the model's behavior through dynamic sigma parameters, enabling refined control over the sampling process.
## Input types
### Required
- **`model`**
    - The model to be enhanced with continuous variable sampling. It serves as the foundation for applying the sampling technique.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`sampling`**
    - Specifies the sampling strategy to be used, which in this case is fixed to 'v_prediction', indicating the use of a prediction-based approach for continuous variable sampling.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sigma_max`**
    - The maximum sigma value for the sampling process, allowing for the adjustment of the sampling's upper bound.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sigma_min`**
    - The minimum sigma value for the sampling process, enabling the setting of the sampling's lower bound.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The enhanced model with the applied continuous variable (V) sampling technique, featuring the integrated specialized sampling mechanism.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ModelSamplingContinuousV:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "sampling": (["v_prediction"],),
                              "sigma_max": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 1000.0, "step":0.001, "round": False}),
                              "sigma_min": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1000.0, "step":0.001, "round": False}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, sampling, sigma_max, sigma_min):
        m = model.clone()

        latent_format = None
        sigma_data = 1.0
        if sampling == "v_prediction":
            sampling_type = comfy.model_sampling.V_PREDICTION

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingContinuousV, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(sigma_min, sigma_max, sigma_data)
        m.add_object_patch("model_sampling", model_sampling)
        return (m, )

```
