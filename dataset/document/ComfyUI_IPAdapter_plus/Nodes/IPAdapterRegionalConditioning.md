---
tags:
- IPAdapter
- RegionalImageProcessing
---

# IPAdapter Regional Conditioning
## Documentation
- Class name: `IPAdapterRegionalConditioning`
- Category: `ipadapter/params`
- Output node: `False`

This node specializes in applying regional conditioning to images within the IPAdapter framework. It leverages masks to selectively apply positive and negative prompts to specific areas of an image, enabling fine-grained control over the generation process. This functionality is crucial for tasks requiring targeted adjustments or enhancements, such as modifying specific image regions without affecting the overall composition.
## Input types
### Required
- **`image`**
    - The image to be conditioned. It serves as the base for applying regional conditioning, determining the areas where positive or negative prompts will be applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`image_weight`**
    - A weight parameter for the image, influencing the degree to which the original image is retained or modified during the conditioning process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `List[float]`
- **`prompt_weight`**
    - The weight assigned to the prompts, dictating the intensity of their influence on the conditioned regions of the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `List[float]`
- **`weight_type`**
    - Specifies the type of weighting mechanism used for conditioning, affecting how image and prompt weights are balanced.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`start_at`**
    - Defines the starting layer for applying conditioning, allowing for layer-specific adjustments.
    - Comfy dtype: `FLOAT`
    - Python dtype: `List[float]`
- **`end_at`**
    - Determines the ending layer for conditioning, marking the depth of influence through the network layers.
    - Comfy dtype: `FLOAT`
    - Python dtype: `List[float]`
### Optional
- **`mask`**
    - An optional mask to define specific regions for conditioning, enabling targeted application of positive or negative prompts.
    - Comfy dtype: `MASK`
    - Python dtype: `Optional[List[torch.Tensor]]`
- **`positive`**
    - Optional positive prompts to be applied to the masked regions, enhancing or adding desired features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `Optional[List[dict]]`
- **`negative`**
    - Optional negative prompts to be applied, used to diminish or remove undesired aspects within the masked areas.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `Optional[List[dict]]`
## Output types
- **`IPADAPTER_PARAMS`**
    - Comfy dtype: `IPADAPTER_PARAMS`
    - The configured parameters for the IPAdapter, ready for integration with the image processing pipeline.
    - Python dtype: `Dict[str, List]`
- **`POSITIVE`**
    - Comfy dtype: `CONDITIONING`
    - The processed positive prompts, adjusted according to the conditioning settings.
    - Python dtype: `Optional[List[dict]]`
- **`NEGATIVE`**
    - Comfy dtype: `CONDITIONING`
    - The processed negative prompts, tailored to the specified conditioning requirements.
    - Python dtype: `Optional[List[dict]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterRegionalConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            #"set_cond_area": (["default", "mask bounds"],),
            "image": ("IMAGE",),
            "image_weight": ("FLOAT", { "default": 1.0, "min": -1.0, "max": 3.0, "step": 0.05 }),
            "prompt_weight": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05 }),
            "weight_type": (WEIGHT_TYPES, ),
            "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
            "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
        }, "optional": {
            "mask": ("MASK",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
        }}

    RETURN_TYPES = ("IPADAPTER_PARAMS", "CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("IPADAPTER_PARAMS", "POSITIVE", "NEGATIVE")
    FUNCTION = "conditioning"

    CATEGORY = "ipadapter/params"

    def conditioning(self, image, image_weight, prompt_weight, weight_type, start_at, end_at, mask=None, positive=None, negative=None):
        set_area_to_bounds = False #if set_cond_area == "default" else True

        if mask is not None:
            if positive is not None:
                positive = conditioning_set_values(positive, {"mask": mask, "set_area_to_bounds": set_area_to_bounds, "mask_strength": prompt_weight})
            if negative is not None:
                negative = conditioning_set_values(negative, {"mask": mask, "set_area_to_bounds": set_area_to_bounds, "mask_strength": prompt_weight})

        ipadapter_params = {
            "image": [image],
            "attn_mask": [mask],
            "weight": [image_weight],
            "weight_type": [weight_type],
            "start_at": [start_at],
            "end_at": [end_at],
        }

        return (ipadapter_params, positive, negative, )

```
