---
tags:
- ImageScaling
---

# PatchModelAddDownscale (Kohya Deep Shrink)
## Documentation
- Class name: `PatchModelAddDownscale`
- Category: `_for_testing`
- Output node: `False`

The `PatchModelAddDownscale` node is designed to modify a given model by integrating downscaling and upscaling processes within its computation flow. This node specifically targets the manipulation of model's internal blocks based on specified parameters, enabling the adjustment of the model's behavior to incorporate downscaling at designated stages of processing. It aims to enhance model efficiency or alter its output characteristics by dynamically adjusting the resolution of intermediate representations.
## Input types
### Required
- **`model`**
    - The model to be patched with downscaling and upscaling functionalities. It serves as the base for modifications applied by the node.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`block_number`**
    - Specifies the block number within the model where the downscaling should be applied, targeting specific stages of the model's processing flow.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`downscale_factor`**
    - The factor by which the model's intermediate representations are downscaled, affecting the resolution and potentially the computational efficiency.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_percent`**
    - Defines the starting percentage of the sigma range for applying downscaling, determining the initial point within the model's processing where downscaling begins.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_percent`**
    - Specifies the ending percentage of the sigma range for downscaling, marking the point within the model's processing where downscaling ceases.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`downscale_after_skip`**
    - A boolean flag indicating whether the downscaling should occur after skip connections within the model, affecting the placement and impact of downscaling operations.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`downscale_method`**
    - The method used for downscaling the model's intermediate representations, influencing the quality and characteristics of the downscaling process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`upscale_method`**
    - The method used for upscaling the model's intermediate representations back to their original resolution, affecting the quality and characteristics of the upscaling process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model with integrated downscaling and upscaling functionalities, reflecting the adjustments made to its internal processing flow.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - KRestartSamplerAdv
    - [ModelSamplingDiscrete](../../Comfy/Nodes/ModelSamplingDiscrete.md)
    - [LoraLoader](../../Comfy/Nodes/LoraLoader.md)
    - IPAdapterApply



## Source code
```python
class PatchModelAddDownscale:
    upscale_methods = ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "block_number": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                              "downscale_factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 9.0, "step": 0.001}),
                              "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                              "end_percent": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001}),
                              "downscale_after_skip": ("BOOLEAN", {"default": True}),
                              "downscale_method": (s.upscale_methods,),
                              "upscale_method": (s.upscale_methods,),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method):
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        def input_block_patch(h, transformer_options):
            if transformer_options["block"][1] == block_number:
                sigma = transformer_options["sigmas"][0].item()
                if sigma <= sigma_start and sigma >= sigma_end:
                    h = comfy.utils.common_upscale(h, round(h.shape[-1] * (1.0 / downscale_factor)), round(h.shape[-2] * (1.0 / downscale_factor)), downscale_method, "disabled")
            return h

        def output_block_patch(h, hsp, transformer_options):
            if h.shape[2] != hsp.shape[2]:
                h = comfy.utils.common_upscale(h, hsp.shape[-1], hsp.shape[-2], upscale_method, "disabled")
            return h, hsp

        m = model.clone()
        if downscale_after_skip:
            m.set_model_input_block_patch_after_skip(input_block_patch)
        else:
            m.set_model_input_block_patch(input_block_patch)
        m.set_model_output_block_patch(output_block_patch)
        return (m, )

```
