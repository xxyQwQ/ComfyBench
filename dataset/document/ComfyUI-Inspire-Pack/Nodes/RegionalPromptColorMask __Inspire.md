---
tags:
- RegionalPrompt
---

# Regional Prompt By Color Mask (Inspire)
## Documentation
- Class name: `RegionalPromptColorMask __Inspire`
- Category: `InspirePack/Regional`
- Output node: `False`

This node specializes in generating regional prompts based on color masks within the Inspire pack. It leverages color masks to apply specific prompts to designated regions of an image, enabling targeted image manipulation or generation.
## Input types
### Required
- **`basic_pipe`**
    - A tuple containing the basic components required for processing, including the model, clip, vae, and positive and negative prompts. It forms the foundation for the prompt generation process.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `tuple`
- **`color_mask`**
    - The color mask image used to define regions for prompt application. It determines the areas of the image that the prompts will specifically target.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask_color`**
    - A string specifying the color in the mask that identifies the target region. This color defines which parts of the image the prompts will be applied to.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`cfg`**
    - A floating-point value representing the configuration setting for the generation process, influencing the level of creativity or adherence to the prompt.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - The name of the sampler used in the generation process, affecting how the model explores the prompt space.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Specifies the scheduler used for controlling the generation process, impacting how the prompt is interpreted over time.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`wildcard_prompt`**
    - A string representing a dynamic or variable part of the prompt that can be filled in with different values for each generation, adding variability to the output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`controlnet_in_pipe`**
    - A boolean indicating whether to keep or override the control net settings in the pipeline, affecting how the model's internal controls are applied.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`sigma_factor`**
    - A floating-point value that adjusts the sigma parameter in the generation process, influencing the diversity of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`variation_seed`**
    - An integer seed for controlling variation in the generation process, ensuring reproducibility of results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`variation_strength`**
    - A floating-point value that determines the strength of the variation applied, affecting the degree of deviation from the base prompt.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`variation_method`**
    - Specifies the method used for applying variation, such as 'linear' or 'slerp', influencing how changes are interpolated.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler_func_opt`**
    - Optional scheduler function options that provide additional control over the scheduling process, allowing for finer adjustments.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `str`
## Output types
- **`regional_prompts`**
    - Comfy dtype: `REGIONAL_PROMPTS`
    - The processed prompt information, tailored for the specified region within the image, ready for further processing or generation tasks.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The output mask derived from the input color mask, indicating the targeted region for the prompt application.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RegionalPromptColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "basic_pipe": ("BASIC_PIPE",),
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (common.SCHEDULERS,),
                "wildcard_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "wildcard prompt"}),
                "controlnet_in_pipe": ("BOOLEAN", {"default": False, "label_on": "Keep", "label_off": "Override"}),
                "sigma_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "variation_method": (["linear", "slerp"],),
                "scheduler_func_opt": ("SCHEDULER_FUNC",),
            }
        }

    RETURN_TYPES = ("REGIONAL_PROMPTS", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(basic_pipe, color_mask, mask_color, cfg, sampler_name, scheduler, wildcard_prompt,
             controlnet_in_pipe=False, sigma_factor=1.0, variation_seed=0, variation_strength=0.0, variation_method="linear", scheduler_func_opt=None):
        mask = color_to_mask(color_mask, mask_color)
        rp = RegionalPromptSimple().doit(basic_pipe, mask, cfg, sampler_name, scheduler, wildcard_prompt, controlnet_in_pipe,
                                         sigma_factor=sigma_factor, variation_seed=variation_seed, variation_strength=variation_strength, variation_method=variation_method, scheduler_func_opt=scheduler_func_opt)[0]
        return rp, mask

```
