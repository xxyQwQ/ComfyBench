---
tags:
- Mask
- MaskGeneration
---

# 🔧 Mask Fix
## Documentation
- Class name: `MaskFix+`
- Category: `essentials/mask`
- Output node: `False`

The MaskFix node is designed to correct or adjust masks within images, focusing on refining mask boundaries, shapes, or overall mask integrity. It aims to enhance the quality and accuracy of masks used in image processing tasks, ensuring they accurately represent the intended areas or objects.
## Input types
### Required
- **`mask`**
    - The 'mask' parameter is the primary input for the MaskFix node, representing the mask to be corrected or adjusted. It plays a crucial role in determining the areas of the image that require refinement.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`erode_dilate`**
    - Specifies the degree to which the mask should be eroded or dilated, affecting the mask's boundaries and overall shape.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`fill_holes`**
    - Determines whether and to what extent holes within the mask should be filled, enhancing mask integrity.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`remove_isolated_pixels`**
    - Controls the removal of isolated pixels, improving the mask's overall appearance and accuracy.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`smooth`**
    - Adjusts the smoothness of the mask's edges, refining its boundaries for a more precise representation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur`**
    - Defines the level of blur applied to the mask, aiding in softening edges and improving visual quality.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a refined or adjusted mask, improved in terms of boundary definition, shape, and overall integrity.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MaskFix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "erode_dilate": ("INT", { "default": 0, "min": -256, "max": 256, "step": 1, }),
                "fill_holes": ("INT", { "default": 0, "min": 0, "max": 128, "step": 1, }),
                "remove_isolated_pixels": ("INT", { "default": 0, "min": 0, "max": 32, "step": 1, }),
                "smooth": ("INT", { "default": 0, "min": 0, "max": 256, "step": 1, }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 256, "step": 1, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, mask, erode_dilate, smooth, remove_isolated_pixels, blur, fill_holes):
        masks = []
        for m in mask:
            # erode and dilate
            if erode_dilate != 0:
                if erode_dilate < 0:
                    m = torch.from_numpy(scipy.ndimage.grey_erosion(m.cpu().numpy(), size=(-erode_dilate, -erode_dilate)))
                else:
                    m = torch.from_numpy(scipy.ndimage.grey_dilation(m.cpu().numpy(), size=(erode_dilate, erode_dilate)))

            # fill holes
            if fill_holes > 0:
                #m = torch.from_numpy(scipy.ndimage.binary_fill_holes(m.cpu().numpy(), structure=np.ones((fill_holes,fill_holes)))).float()
                m = torch.from_numpy(scipy.ndimage.grey_closing(m.cpu().numpy(), size=(fill_holes, fill_holes)))

            # remove isolated pixels
            if remove_isolated_pixels > 0:
                m = torch.from_numpy(scipy.ndimage.grey_opening(m.cpu().numpy(), size=(remove_isolated_pixels, remove_isolated_pixels)))

            # smooth the mask
            if smooth > 0:
                if smooth % 2 == 0:
                    smooth += 1
                m = T.functional.gaussian_blur((m > 0.5).unsqueeze(0), smooth).squeeze(0)

            # blur the mask
            if blur > 0:
                if blur % 2 == 0:
                    blur += 1
                m = T.functional.gaussian_blur(m.float().unsqueeze(0), blur).squeeze(0)

            masks.append(m.float())

        masks = torch.stack(masks, dim=0).float()

        return (masks, )

```
