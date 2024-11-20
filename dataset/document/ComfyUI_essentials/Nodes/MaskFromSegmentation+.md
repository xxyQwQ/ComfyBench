---
tags:
- Segmentation
---

# 🔧 Mask From Segmentation
## Documentation
- Class name: `MaskFromSegmentation+`
- Category: `essentials/mask`
- Output node: `False`

This node is designed to generate masks from image segmentation, effectively converting segmented areas into distinct mask layers. It allows for advanced image editing and manipulation by isolating specific segments of an image for targeted operations.
## Input types
### Required
- **`image`**
    - The input image to be segmented into masks. It serves as the basis for generating distinct mask layers for each segmented area.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`segments`**
    - The number of segments to divide the image into. This parameter controls the granularity of the segmentation, with higher values resulting in more detailed masks.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`remove_isolated_pixels`**
    - A threshold for removing isolated pixels within a mask to clean up noise and improve mask quality.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`remove_small_masks`**
    - A threshold for removing small masks deemed to be noise, ensuring only significant segments are retained.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`fill_holes`**
    - A boolean flag indicating whether to fill holes within the generated masks, enhancing their completeness and usability.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output consists of a stack of masks corresponding to the segmented areas of the input image, ready for further processing or application.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MaskFromSegmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "segments": ("INT", { "default": 6, "min": 1, "max": 16, "step": 1, }),
                "remove_isolated_pixels": ("INT", { "default": 0, "min": 0, "max": 32, "step": 1, }),
                "remove_small_masks": ("FLOAT", { "default": 0.0, "min": 0., "max": 1., "step": 0.01, }),
                "fill_holes": ("BOOLEAN", { "default": False }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, image, segments, remove_isolated_pixels, fill_holes, remove_small_masks):
        im = image[0] # we only work on the first image in the batch
        im = Image.fromarray((im * 255).to(torch.uint8).cpu().numpy(), mode="RGB")
        im = im.quantize(palette=im.quantize(colors=segments), dither=Image.Dither.NONE)
        im = torch.tensor(np.array(im.convert("RGB"))).float() / 255.0

        colors = im.reshape(-1, im.shape[-1])
        colors = torch.unique(colors, dim=0)

        masks = []
        for color in colors:
            mask = (im == color).all(dim=-1).float()
            # remove isolated pixels
            if remove_isolated_pixels > 0:
                mask = torch.from_numpy(scipy.ndimage.binary_opening(mask.cpu().numpy(), structure=np.ones((remove_isolated_pixels, remove_isolated_pixels))))

            # fill holes
            if fill_holes:
                mask = torch.from_numpy(scipy.ndimage.binary_fill_holes(mask.cpu().numpy()))

            # if the mask is too small, it's probably noise
            if mask.sum() / (mask.shape[0]*mask.shape[1]) > remove_small_masks:
                masks.append(mask)

        if masks == []:
            masks.append(torch.zeros_like(im)[:,:,0]) # return an empty mask if no masks were found, prevents errors

        mask = torch.stack(masks, dim=0).float()

        return (mask, )

```
