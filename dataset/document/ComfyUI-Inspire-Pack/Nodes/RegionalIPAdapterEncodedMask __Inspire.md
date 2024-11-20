---
tags:
- IPAdapter
- RegionalImageProcessing
---

# Regional IPAdapter Encoded Mask (Inspire)
## Documentation
- Class name: `RegionalIPAdapterEncodedMask __Inspire`
- Category: `InspirePack/Regional`
- Output node: `False`

This node specializes in applying encoded mask-based image processing adaptations to enhance or modify images, leveraging regional IP adapter techniques. It allows for precise control over the image adaptation process through the use of masks, embedding weights, and various parameters, enabling customized image outputs tailored to specific regions of interest.
## Input types
### Required
- **`mask`**
    - The mask parameter specifies the area of the image to be adapted, serving as a crucial element in defining the scope of the image processing operation.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`embeds`**
    - Embeds represent the encoded information or features to be applied to the specified image regions, playing a pivotal role in the adaptation process.
    - Comfy dtype: `EMBEDS`
    - Python dtype: `torch.Tensor`
- **`weight`**
    - Weight determines the intensity or influence of the applied adaptations, allowing for fine-tuning of the image processing effects.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_type`**
    - Weight type specifies the method of applying weights, offering options like original, linear, or channel penalty to customize the adaptation effect.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_at`**
    - Start at indicates the initial point in the adaptation process, enabling phased or gradual application of changes.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - End at defines the conclusion point for the adaptations, allowing for precise control over the extent of image processing.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`unfold_batch`**
    - Unfold batch toggles the processing of images in batches, potentially optimizing performance for bulk adaptations.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`neg_embeds`**
    - Neg embeds allow for the application of inverse or contrasting features to the specified regions, enhancing the adaptability of the process.
    - Comfy dtype: `EMBEDS`
    - Python dtype: `torch.Tensor`
## Output types
- **`regional_ipadapter`**
    - Comfy dtype: `REGIONAL_IPADAPTER`
    - Outputs a customized image with applied regional adaptations, reflecting the encoded mask and parameter specifications.
    - Python dtype: `CustomType`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RegionalIPAdapterEncodedMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),

                "embeds": ("EMBEDS",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "weight_type": (["original", "linear", "channel penalty"],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "unfold_batch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "neg_embeds": ("EMBEDS",),
            }
        }

    RETURN_TYPES = ("REGIONAL_IPADAPTER", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(mask, embeds, weight, weight_type, start_at=0.0, end_at=1.0, unfold_batch=False, neg_embeds=None):
        cond = IPAdapterConditioning(mask, weight, weight_type, embeds=embeds, start_at=start_at, end_at=end_at, unfold_batch=unfold_batch, neg_embeds=neg_embeds)
        return (cond, )

```
