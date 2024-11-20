---
tags:
- Sampling
---

# FreeU_V2
## Documentation
- Class name: `FreeU_V2`
- Category: `model_patches`
- Output node: `False`

The FreeU_V2 node enhances the functionality of generative models by applying dynamic scaling and filtering techniques to the model's output. It utilizes a scale dictionary to adjust the model's hidden states based on predefined scaling factors and applies a Fourier filter to the spatial components, ensuring optimized output quality. This node is particularly useful for improving the visual fidelity of generated images or patterns, making it a valuable tool for tasks requiring high-quality visual outputs.
## Input types
### Required
- **`model`**
    - The generative model to be enhanced by the FreeU_V2 node. It is crucial for defining the base functionality that will be augmented by the node's scaling and filtering operations.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`b1`**
    - A scaling factor for adjusting the model's hidden states, contributing to the dynamic scaling functionality of the node.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`b2`**
    - Another scaling factor for adjusting the model's hidden states, working alongside b1 to fine-tune the output quality.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`s1`**
    - A scaling parameter used in the Fourier filtering process to modify the spatial components of the model's output, enhancing visual clarity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`s2`**
    - A secondary scaling parameter for the Fourier filter, used to further refine the spatial aspects of the generated output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The enhanced generative model, with improved output quality through dynamic scaling and Fourier filtering.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [UltimateSDUpscale](../../ComfyUI_UltimateSDUpscale/Nodes/UltimateSDUpscale.md)
    - [FaceDetailer](../../ComfyUI-Impact-Pack/Nodes/FaceDetailer.md)
    - [Anything Everywhere](../../cg-use-everywhere/Nodes/Anything Everywhere.md)
    - Reroute
    - [PatchModelAddDownscale](../../Comfy/Nodes/PatchModelAddDownscale.md)
    - [KSamplerAdvanced](../../Comfy/Nodes/KSamplerAdvanced.md)
    - [VideoLinearCFGGuidance](../../Comfy/Nodes/VideoLinearCFGGuidance.md)
    - [IPAdapter](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapter.md)
    - [UltimateSDUpscaleNoUpscale](../../ComfyUI_UltimateSDUpscale/Nodes/UltimateSDUpscaleNoUpscale.md)



## Source code
```python
class FreeU_V2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "b1": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "b2": ("FLOAT", {"default": 1.4, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "s1": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "s2": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, b1, b2, s1, s2):
        model_channels = model.model.model_config.unet_config["model_channels"]
        scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
        on_cpu_devices = {}

        def output_block_patch(h, hsp, transformer_options):
            scale = scale_dict.get(int(h.shape[1]), None)
            if scale is not None:
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                h[:,:h.shape[1] // 2] = h[:,:h.shape[1] // 2] * ((scale[0] - 1 ) * hidden_mean + 1)

                if hsp.device not in on_cpu_devices:
                    try:
                        hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
                    except:
                        logging.warning("Device {} does not support the torch.fft functions used in the FreeU node, switching to CPU.".format(hsp.device))
                        on_cpu_devices[hsp.device] = True
                        hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)
                else:
                    hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)

            return h, hsp

        m = model.clone()
        m.set_model_output_block_patch(output_block_patch)
        return (m, )

```
