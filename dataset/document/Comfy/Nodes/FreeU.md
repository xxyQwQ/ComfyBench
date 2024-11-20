---
tags:
- Sampling
---

# FreeU
## Documentation
- Class name: `FreeU`
- Category: `model_patches`
- Output node: `False`

The FreeU node is designed to enhance the functionality of neural network models by applying a dynamic scaling technique to the model's hidden layers. This process involves adjusting the scaling of specific layers based on predefined criteria, potentially improving the model's performance or efficiency. Additionally, the node incorporates a mechanism to handle operations on devices that do not support certain functions, gracefully degrading to CPU processing when necessary.
## Input types
### Required
- **`model`**
    - The model to be enhanced with the FreeU node's dynamic scaling technique. It affects the node's execution by determining which layers are scaled and how, ultimately influencing the model's performance.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`b1`**
    - A scaling factor for the model's hidden layers, contributing to the dynamic adjustment of the model's architecture for improved performance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`b2`**
    - Another scaling factor for the model's hidden layers, working alongside b1 to fine-tune the model's scaling for optimal performance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`s1`**
    - A parameter influencing the scaling process within the FreeU node, affecting the model's efficiency and output quality.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`s2`**
    - Complements s1 as part of the FreeU node's scaling mechanism, further refining the model's performance through dynamic layer adjustment.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The enhanced model, after applying the FreeU node's dynamic scaling technique. This output reflects the adjustments made to the model's architecture for improved performance or efficiency.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [KSamplerAdvanced](../../Comfy/Nodes/KSamplerAdvanced.md)
    - [UltimateSDUpscale](../../ComfyUI_UltimateSDUpscale/Nodes/UltimateSDUpscale.md)
    - [Bus Node](../../was-node-suite-comfyui/Nodes/Bus Node.md)
    - DynamicThresholdingFull
    - [Anything Everywhere](../../cg-use-everywhere/Nodes/Anything Everywhere.md)



## Source code
```python
class FreeU:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "b1": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "b2": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.01}),
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
                h[:,:h.shape[1] // 2] = h[:,:h.shape[1] // 2] * scale[0]
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
