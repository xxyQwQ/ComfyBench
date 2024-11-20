---
tags:
- ControlNet
---

# ControlNetApplySD3
## Documentation
- Class name: `ControlNetApplySD3`
- Category: `_for_testing/sd3`
- Output node: `False`

The ControlNetApplySD3 node is designed for applying a control network to modify the generation of images based on positive and negative conditioning inputs. It leverages a control network, a VAE, and specific strength parameters to influence the image generation process, allowing for fine-tuned control over the output based on the provided conditioning.
## Input types
### Required
- **`positive`**
    - Positive conditioning inputs that guide the image generation towards desired attributes or features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `tuple`
- **`negative`**
    - Negative conditioning inputs that guide the image generation away from certain attributes or features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `tuple`
- **`control_net`**
    - The control network used to apply specific modifications or influences on the image generation process.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `ControlNetType`
- **`vae`**
    - A variational autoencoder used for encoding and decoding images, contributing to the control network's ability to modify the image generation.
    - Comfy dtype: `VAE`
    - Python dtype: `VAEType`
- **`image`**
    - The input image to be modified by the control network based on the conditioning.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`strength`**
    - A parameter controlling the intensity of the control network's influence on the image generation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_percent`**
    - The starting percentage of the control effect applied to the image generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_percent`**
    - The ending percentage of the control effect applied to the image generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - The modified positive conditioning after applying the control network.
    - Python dtype: `tuple`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - The modified negative conditioning after applying the control network.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ControlNetApplySD3(nodes.ControlNetApplyAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "vae": ("VAE", ),
                             "image": ("IMAGE", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                             }}
    CATEGORY = "_for_testing/sd3"

```
