---
tags:
- RegionalPrompt
---

# Regional Prompt Simple (Inspire)
## Documentation
- Class name: `RegionalPromptSimple __Inspire`
- Category: `InspirePack/Regional`
- Output node: `False`

The RegionalPromptSimple node is designed to facilitate the creation of region-specific prompts within an image generation pipeline. It leverages regional information to tailor the generation process, ensuring that the output images are aligned with the specified regional characteristics and constraints.
## Input types
### Required
- **`basic_pipe`**
    - Represents the foundational components of the image generation pipeline, including models and initial conditions. It's essential for the node to function within the broader generation context.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `Tuple[Model, ClipModel, VAEModel, List[Tuple[str, Dict[str, Any]]], List[Tuple[str, Dict[str, Any]]]]`
- **`mask`**
    - An image mask defining the regions of interest. It is used to apply regional prompts specifically to areas highlighted by the mask.
    - Comfy dtype: `MASK`
    - Python dtype: `Image`
- **`cfg`**
    - Controls the conditioning factor for generation, affecting how strongly the regional prompts influence the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Determines the sampling method used during image generation, impacting the diversity and quality of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Defines the scheduling algorithm for controlling the generation process, influencing the progression and variation of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`wildcard_prompt`**
    - Allows for the inclusion of dynamic or variable text elements within the prompts, adding flexibility to the generation process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`controlnet_in_pipe`**
    - Indicates whether control networks are included in the pipeline, affecting the application of regional prompts.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`sigma_factor`**
    - Adjusts the noise level in the generation process, influencing the variability and creativity of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`variation_seed`**
    - A seed for introducing variations in the generation process, providing a way to explore different outcomes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`variation_strength`**
    - Determines the strength of the variations introduced by the variation seed, affecting the diversity of the generated output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`variation_method`**
    - Specifies the method used for applying variations, influencing how changes are integrated into the generation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler_func_opt`**
    - An optional scheduling function that can be used to further customize the generation process.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `SchedulerFunc`
## Output types
- **`regional_prompts`**
    - Comfy dtype: `REGIONAL_PROMPTS`
    - The generated regional prompts, tailored to specific areas of the image as defined by the input mask.
    - Python dtype: `List[RegionalPrompt]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RegionalPromptSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "basic_pipe": ("BASIC_PIPE",),
                "mask": ("MASK",),
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

    RETURN_TYPES = ("REGIONAL_PROMPTS", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(basic_pipe, mask, cfg, sampler_name, scheduler, wildcard_prompt,
             controlnet_in_pipe=False, sigma_factor=1.0, variation_seed=0, variation_strength=0.0, variation_method='linear', scheduler_func_opt=None):
        if 'RegionalPrompt' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/ltdrdata/ComfyUI-Impact-Pack',
                                          "To use 'RegionalPromptSimple' node, 'Impact Pack' extension is required.")
            raise Exception(f"[ERROR] To use RegionalPromptSimple, you need to install 'ComfyUI-Impact-Pack'")

        model, clip, vae, positive, negative = basic_pipe

        iwe = nodes.NODE_CLASS_MAPPINGS['ImpactWildcardEncode']()
        kap = nodes.NODE_CLASS_MAPPINGS['KSamplerAdvancedProvider']()
        rp = nodes.NODE_CLASS_MAPPINGS['RegionalPrompt']()

        if wildcard_prompt != "":
            model, clip, new_positive, _ = iwe.doit(model=model, clip=clip, populated_text=wildcard_prompt, seed=None)

            if controlnet_in_pipe:
                prev_cnet = None
                for t in positive:
                    if 'control' in t[1] and 'control_apply_to_uncond' in t[1]:
                        prev_cnet = t[1]['control'], t[1]['control_apply_to_uncond']
                        break

                if prev_cnet is not None:
                    for t in new_positive:
                        t[1]['control'] = prev_cnet[0]
                        t[1]['control_apply_to_uncond'] = prev_cnet[1]

        else:
            new_positive = positive

        basic_pipe = model, clip, vae, new_positive, negative

        sampler = kap.doit(cfg, sampler_name, scheduler, basic_pipe, sigma_factor=sigma_factor, scheduler_func_opt=scheduler_func_opt)[0]
        try:
            regional_prompts = rp.doit(mask, sampler, variation_seed=variation_seed, variation_strength=variation_strength, variation_method=variation_method)[0]
        except:
            raise Exception("[Inspire-Pack] ERROR: Impact Pack is outdated. Update Impact Pack to latest version to use this.")

        return (regional_prompts, )

```
