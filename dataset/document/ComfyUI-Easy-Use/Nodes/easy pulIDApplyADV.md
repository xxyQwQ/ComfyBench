---
tags:
- IdentityImage
---

# Easy Apply PuLID (Advanced)
## Documentation
- Class name: `easy pulIDApplyADV`
- Category: `EasyUse/Adapter`
- Output node: `False`

The `easy pulIDApplyADV` node is designed to enhance images by applying advanced PulID transformations. It leverages PulID models, InsightFace for facial analysis, and optionally EVA Clip for additional enhancements. This node is capable of adjusting images based on various parameters such as weight, projection, fidelity, and noise, offering a flexible approach to image manipulation.
## Input types
### Required
- **`model`**
    - The model input is the base model onto which the PulID transformations will be applied. It is central to the node's operation, serving as the foundation for the enhancements.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`pulid_file`**
    - Specifies the PulID file to be used for the transformation. It is crucial for defining the specific PulID model that will be applied to the input model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`insightface`**
    - Determines the InsightFace model to be used for facial analysis, supporting different computation backends. It plays a key role in facial feature detection and analysis within the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`image`**
    - The input image to be enhanced. It is directly affected by the PulID transformations, with changes being applied based on the specified parameters.
    - Comfy dtype: `IMAGE`
    - Python dtype: `PIL.Image.Image`
- **`weight`**
    - Controls the intensity of the PulID transformation applied to the image. It allows for fine-tuning the impact of the transformation on the final output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`projection`**
    - Defines the projection method used in the transformation process. It influences how the PulID model interprets and applies enhancements to the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`fidelity`**
    - Adjusts the fidelity of the transformation, affecting the accuracy and detail preservation in the enhanced image.
    - Comfy dtype: `INT`
    - Python dtype: `str`
- **`noise`**
    - Specifies the level of noise to be added to the transformation. It can be used to introduce variability or texture into the enhanced image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_at`**
    - Determines the starting point of the transformation within the image. It allows for targeted application of the PulID enhancements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - Specifies the ending point of the transformation within the image, enabling precise control over the area affected by the PulID enhancements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`attn_mask`**
    - An optional attention mask that can be applied to focus the PulID transformation on specific areas of the image.
    - Comfy dtype: `MASK`
    - Python dtype: `numpy.ndarray`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The enhanced model output, which has undergone the PulID transformation. It reflects the applied enhancements and adjustments.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class applyPulIDADV(applyPulID):

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "pulid_file": (folder_paths.get_filename_list("pulid"),),
                "insightface": (["CPU", "CUDA", "ROCM"],),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "projection": (["ortho_v2", "ortho", "none"], {"default":"ortho_v2"}),
                "fidelity": ("INT", {"default": 8, "min": 0, "max": 32, "step": 1}),
                "noise": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "attn_mask": ("MASK",),
            },
        }

```
