---
tags:
- Sampling
---

# KSampler Advanced Progress (Inspire)
## Documentation
- Class name: `KSamplerAdvancedProgress __Inspire`
- Category: `InspirePack/analysis`
- Output node: `False`

This node represents an advanced version of the KSampler for progressive sampling in the Inspire pack. It extends the functionality of the standard KSampler by incorporating advanced features for more nuanced control over the sampling process, including progress tracking and optional callbacks for intermediate steps. This allows for a more detailed and customizable generation of samples, catering to specific needs and enhancing the overall sampling experience.
## Input types
### Required
- **`model`**
    - The model parameter represents the generative model to be used for sampling. It is crucial for defining the behavior and output of the sampling process.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`add_noise`**
    - Indicates whether noise should be added to the sampling process, affecting the diversity and quality of the generated samples.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`noise_seed`**
    - The seed for noise generation, ensuring reproducibility and control over the randomness introduced in the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the number of steps to be taken in the sampling process, directly impacting the detail and quality of the generated samples.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration settings for the sampling process, allowing for customization of various aspects such as temperature and noise levels.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Specifies the particular sampler to be used, enabling the selection of different sampling strategies based on the task at hand.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - The scheduler parameter controls the progression of sampling steps, allowing for fine-tuned adjustments to the sampling dynamics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `object`
- **`positive`**
    - A list of positive prompts guiding the sampling towards desired attributes or themes in the generated samples.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `list[str]`
- **`negative`**
    - A list of negative prompts used to steer the sampling away from undesired attributes or themes, enhancing the relevance of the output.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `list[str]`
- **`latent_image`**
    - The initial latent image to start the sampling from, setting the baseline for the generation process.
    - Comfy dtype: `LATENT`
    - Python dtype: `dict`
- **`start_at_step`**
    - Specifies the starting step for the sampling process, allowing for resumed or phased sampling operations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at_step`**
    - Defines the end step of the sampling process, marking the completion of the generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`noise_mode`**
    - Determines the mode of noise application during the sampling, influencing the texture and variability of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`return_with_leftover_noise`**
    - Indicates whether the final output should include residual noise, offering an option for post-processing or further manipulation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`interval`**
    - The interval at which progress callbacks are triggered, enabling periodic updates and adjustments during the sampling.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`omit_start_latent`**
    - A flag to exclude the starting latent image from the results, useful for focusing on the changes introduced during the sampling.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`prev_progress_latent_opt`**
    - unknown
    - Comfy dtype: `LATENT`
    - Python dtype: `unknown`
- **`scheduler_func_opt`**
    - An optional scheduler function for additional control over the sampling steps, enhancing the flexibility of the process.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `function`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - unknown
    - Python dtype: `unknown`
- **`progress_latent`**
    - Comfy dtype: `LATENT`
    - The final generated samples or progress states, encapsulating the outcomes of the advanced sampling process.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class KSamplerAdvanced_progress(a1111_compat.KSamplerAdvanced_inspire):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "add_noise": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (common.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "noise_mode": (["GPU(=A1111)", "CPU"],),
                    "return_with_leftover_noise": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
                    "interval": ("INT", {"default": 1, "min": 1, "max": 10000}),
                    "omit_start_latent": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                    },
                "optional": {
                    "prev_progress_latent_opt": ("LATENT",),
                    "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    }
                }

    FUNCTION = "doit"

    CATEGORY = "InspirePack/analysis"

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("latent", "progress_latent")

    def doit(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step,
             noise_mode, return_with_leftover_noise, interval, omit_start_latent, prev_progress_latent_opt=None, scheduler_func_opt=None):
        if omit_start_latent:
            result = []
        else:
            result = [latent_image['samples']]

        result = []

        def progress_callback(step, x0, x, total_steps):
            if (total_steps-1) != step and step % interval != 0:
                return

            x = model.model.process_latent_out(x)
            x = x.to(model_management.intermediate_device())
            result.append(x)

        latent_image, noise = a1111_compat.KSamplerAdvanced_inspire.sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step,
                                                                           noise_mode, False, callback=progress_callback, scheduler_func_opt=scheduler_func_opt)

        if len(result) > 0:
            result = torch.cat(result)
            result = {'samples': result}
        else:
            result = latent_image

        if prev_progress_latent_opt is not None:
            result['samples'] = torch.cat((prev_progress_latent_opt['samples'], result['samples']), dim=0)

        return latent_image, result

```
