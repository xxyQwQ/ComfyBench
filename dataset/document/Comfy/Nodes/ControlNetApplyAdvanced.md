---
tags:
- ControlNet
---

# Apply ControlNet (Advanced)
## Documentation
- Class name: `ControlNetApplyAdvanced`
- Category: `conditioning`
- Output node: `False`

This node is designed for advanced application of a control network to modify the conditioning of an image generation process. It leverages a control network to adjust the conditioning based on the provided image, strength, and additional parameters, enhancing the control over the generation outcome.
## Input types
### Required
- **`positive`**
    - The positive conditioning data that the control network will modify, representing the desired attributes or features to emphasize in the generated image.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `list of tuples`
- **`negative`**
    - The negative conditioning data that the control network will modify, representing the attributes or features to de-emphasize or avoid in the generated image.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `list of tuples`
- **`control_net`**
    - The control network model used to apply modifications to the conditioning. It is central to the node's functionality, enabling the dynamic adjustment of the generation process.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `ControlNet`
- **`image`**
    - The image data that influences the control network's modifications to the conditioning. It serves as a contextual basis for the adjustments.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`strength`**
    - A scalar value that determines the intensity of the control network's modifications. It allows for fine-tuning the impact of the control network on the conditioning.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_percent`**
    - The starting percentage of the control effect, allowing for gradual application of the control network's influence over the image generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_percent`**
    - The ending percentage of the control effect, defining the point at which the control network's influence ceases, enabling precise control over the conditioning modification.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - The modified positive conditioning data after the control network's application, reflecting the desired adjustments.
    - Python dtype: `list of tuples`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - The modified negative conditioning data after the control network's application, reflecting the de-emphasized or avoided attributes.
    - Python dtype: `list of tuples`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [KSamplerAdvanced](../../Comfy/Nodes/KSamplerAdvanced.md)
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)
    - KSampler //Inspire
    - [UltimateSDUpscale](../../ComfyUI_UltimateSDUpscale/Nodes/UltimateSDUpscale.md)
    - [ToBasicPipe](../../ComfyUI-Impact-Pack/Nodes/ToBasicPipe.md)
    - [FaceDetailer](../../ComfyUI-Impact-Pack/Nodes/FaceDetailer.md)
    - Reroute
    - [SamplerCustom](../../Comfy/Nodes/SamplerCustom.md)
    - [Bus Node](../../was-node-suite-comfyui/Nodes/Bus Node.md)



## Source code
```python
class ControlNetApplyAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "image": ("IMAGE", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                             }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "conditioning"

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent, vae=None):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])

```
