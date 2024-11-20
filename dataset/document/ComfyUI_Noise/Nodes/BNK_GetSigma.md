---
tags:
- SigmaScheduling
---

# Get Sigma
## Documentation
- Class name: `BNK_GetSigma`
- Category: `latent/noise`
- Output node: `False`

The `BNK_GetSigma` node is designed to calculate a specific sigma value based on the provided model, sampler name, scheduler, and step parameters. It focuses on determining the appropriate noise level for latent space manipulation, facilitating the generation of variations or adjustments in generated images or data. This node plays a crucial role in controlling the noise injection process, which is essential for fine-tuning the details and quality of generative models' outputs.
## Input types
### Required
- **`model`**
    - The `model` parameter specifies the generative model to be used for calculating sigma. It is crucial for determining the appropriate noise level based on the model's characteristics and latent space.
    - Comfy dtype: `MODEL`
    - Python dtype: `object`
- **`sampler_name`**
    - The `sampler_name` parameter identifies the sampling strategy to be employed. It affects how the noise levels are calculated and applied, influencing the variation and quality of the generated outputs.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `comfy.samplers.KSampler.SAMPLERS`
- **`scheduler`**
    - The `scheduler` parameter defines the scheduling strategy for noise level adjustment over steps. It plays a key role in the temporal dynamics of noise injection, affecting the smoothness and progression of generated variations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `comfy.samplers.KSampler.SCHEDULERS`
- **`steps`**
    - Specifies the total number of steps for the noise calculation process. It determines the granularity and range of sigma values to be considered.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`start_at_step`**
    - Defines the starting step for sigma calculation, allowing for targeted noise level adjustments from a specific point in the generation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at_step`**
    - Marks the ending step for sigma calculation, enabling precise control over the extent of noise adjustments in the generative process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The output is a floating-point number representing the calculated sigma value. This value is crucial for adjusting the noise level in the latent space, directly influencing the detail and quality of the generated outputs.
    - Python dtype: `float`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GetSigma:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            "steps": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "end_at_step": ("INT", {"default": 10000, "min": 1, "max": 10000}),
            }}
    
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "calc_sigma"

    CATEGORY = "latent/noise"
        
    def calc_sigma(self, model, sampler_name, scheduler, steps, start_at_step, end_at_step):
        device = comfy.model_management.get_torch_device()
        end_at_step = min(steps, end_at_step)
        start_at_step = min(start_at_step, end_at_step)
        comfy.model_management.load_model_gpu(model)
        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas
        sigma = sigmas[start_at_step] - sigmas[end_at_step]
        sigma /= model.model.latent_format.scale_factor
        return (sigma.cpu().numpy(),)

```
