---
tags:
- IPAdapter
- RegionalImageProcessing
---

# Regional IPAdapter Encoded By Color Mask (Inspire)
## Documentation
- Class name: `RegionalIPAdapterEncodedColorMask __Inspire`
- Category: `InspirePack/Regional`
- Output node: `False`

This node specializes in applying regional image processing adaptations based on encoded color masks, enabling targeted image manipulation and enhancement. It leverages color masks to conditionally apply transformations or effects to specific regions of an image, guided by embedding weights and types.
## Input types
### Required
- **`color_mask`**
    - The color mask image used to identify specific regions for adaptation. It serves as a key input for determining where the encoded adjustments should be applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask_color`**
    - A string specifying the color used in the color mask to identify the target region. This color determines which parts of the image will be affected by the regional adaptations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`embeds`**
    - Embeddings representing the desired adjustments or effects to be applied to the identified regions. These embeddings guide the adaptation process.
    - Comfy dtype: `EMBEDS`
    - Python dtype: `torch.Tensor`
- **`weight`**
    - A floating-point value that adjusts the intensity or influence of the applied embeddings on the target regions.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_type`**
    - Specifies the method of applying weights to the embeddings, such as linear or channel-specific penalties, influencing how the adaptations are blended into the target regions.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`start_at`**
    - Defines the starting point of the effect's intensity gradient, allowing for gradual application of the adaptation from this point onwards.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - Sets the endpoint for the effect's intensity gradient, ensuring the adaptation effect tapers off at this specified point.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`unfold_batch`**
    - A boolean indicating whether to process each item in a batch separately, affecting the adaptation's application across multiple images.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`neg_embeds`**
    - Optional embeddings that represent negative adjustments or effects, providing a means to counterbalance or negate certain aspects of the primary embeddings.
    - Comfy dtype: `EMBEDS`
    - Python dtype: `torch.Tensor`
## Output types
- **`regional_ipadapter`**
    - Comfy dtype: `REGIONAL_IPADAPTER`
    - The result of applying the regional IP adapter process, encapsulating the conditioned adaptations based on the input parameters.
    - Python dtype: `IPAdapterConditioning`
- **`mask`**
    - Comfy dtype: `MASK`
    - The processed mask derived from the input color mask and mask color, indicating the regions of the image that were targeted for adaptation.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RegionalIPAdapterEncodedColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),

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

    RETURN_TYPES = ("REGIONAL_IPADAPTER", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(color_mask, mask_color, embeds, weight, weight_type, start_at=0.0, end_at=1.0, unfold_batch=False, neg_embeds=None):
        mask = color_to_mask(color_mask, mask_color)
        cond = IPAdapterConditioning(mask, weight, weight_type, embeds=embeds, start_at=start_at, end_at=end_at, unfold_batch=unfold_batch, neg_embeds=neg_embeds)
        return (cond, mask)

```
