---
tags:
- IPAdapter
- RegionalImageProcessing
---

# Regional IPAdapter By Color Mask (Inspire)
## Documentation
- Class name: `RegionalIPAdapterColorMask __Inspire`
- Category: `InspirePack/Regional`
- Output node: `False`

This node is designed to adapt image processing inputs regionally by applying a color mask. It allows for the selective application of image embeddings and weights based on specified color regions within an image, facilitating targeted image manipulation and enhancement.
## Input types
### Required
- **`color_mask`**
    - The color mask image used to define regions for applying the image processing. It serves as a spatial filter to apply different processing techniques to specific areas of an image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask_color`**
    - The specific color value used to identify the regions of interest within the color mask. This color defines which parts of the image will be subject to the specified image processing operations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`image`**
    - The image to which the regional adaptation and processing will be applied, using the defined color mask to target specific areas.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`weight`**
    - A scalar value that adjusts the intensity or influence of the applied embeddings on the selected regions. It modulates how strongly the specified image features are applied.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`noise`**
    - A scalar value representing the amount of noise to be applied within the specified regions, affecting the texture and overall appearance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_type`**
    - Specifies the method of applying weights to the embeddings, such as linear scaling or channel-specific penalties, offering control over the modification intensity.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list`
- **`start_at`**
    - Defines the starting point of the effect's intensity gradient, allowing for gradual application of the image processing from a certain threshold.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - Sets the endpoint for the effect's intensity gradient, enabling a smooth transition of the image processing effect across the specified region.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`unfold_batch`**
    - A boolean flag that determines whether the batch of images should be processed individually or as a group, affecting the computational efficiency and result granularity.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`faceid_v2`**
    - A boolean flag indicating whether to use an advanced face identification method for more precise region targeting within the image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`weight_v2`**
    - An optional scalar value that provides an alternative weighting mechanism for the applied embeddings, offering additional control over the intensity of the image processing.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`combine_embeds`**
    - Specifies the method for combining multiple embeddings, such as concatenation or averaging, to achieve the desired effect within the targeted regions.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`neg_image`**
    - An optional image representing negative features or styles to be subtracted from the specified regions, allowing for more nuanced image manipulations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`regional_ipadapter`**
    - Comfy dtype: `REGIONAL_IPADAPTER`
    - The adapted image processing conditions tailored to the specified regions, ready for further image synthesis or manipulation tasks.
    - Python dtype: `IPAdapterConditioning`
- **`mask`**
    - Comfy dtype: `MASK`
    - The generated mask based on the specified color, indicating the regions of the image that were targeted for processing.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RegionalIPAdapterColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "noise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_type": (["original", "linear", "channel penalty"], ),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "unfold_batch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "faceid_v2": ("BOOLEAN", {"default": False }),
                "weight_v2": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "neg_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("REGIONAL_IPADAPTER", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(color_mask, mask_color, image, weight, noise, weight_type, start_at=0.0, end_at=1.0, unfold_batch=False, faceid_v2=False, weight_v2=False, combine_embeds="concat", neg_image=None):
        mask = color_to_mask(color_mask, mask_color)
        cond = IPAdapterConditioning(mask, weight, weight_type, noise=noise, image=image, neg_image=neg_image, start_at=start_at, end_at=end_at, unfold_batch=unfold_batch, weight_v2=weight_v2, combine_embeds=combine_embeds)
        return (cond, mask)

```
