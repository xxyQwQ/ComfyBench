---
tags:
- Image
- ImageBlend
- ImageComposite
---

# 🔧 Image Composite From Mask Batch
## Documentation
- Class name: `ImageCompositeFromMaskBatch+`
- Category: `essentials/image manipulation`
- Output node: `False`

This node is designed for image manipulation tasks, specifically for creating composite images from two source images based on a mask. It blends parts of the 'image_from' and 'image_to' images according to the mask, allowing for sophisticated image editing and composition techniques.
## Input types
### Required
- **`image_from`**
    - The source image from which pixels are taken when the mask is not applied. It plays a crucial role in determining the final composite image's appearance.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image_to`**
    - The target image to which pixels are added based on the mask. It significantly influences the outcome of the composite image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - A binary or grayscale mask that determines how pixels from 'image_from' and 'image_to' are blended together. The mask's values dictate the blending process, affecting the composite image's visual result.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting composite image, created by blending 'image_from' and 'image_to' according to the mask. It showcases the combined visual elements of both source images.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageCompositeFromMaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_from": ("IMAGE", ),
                "image_to": ("IMAGE", ),
                "mask": ("MASK", )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image_from, image_to, mask):
        frames = mask.shape[0]

        if image_from.shape[1] != image_to.shape[1] or image_from.shape[2] != image_to.shape[2]:
            image_to = comfy.utils.common_upscale(image_to.permute([0,3,1,2]), image_from.shape[2], image_from.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])

        if frames < image_from.shape[0]:
            image_from = image_from[:frames]
        elif frames > image_from.shape[0]:
            image_from = torch.cat((image_from, image_from[-1].unsqueeze(0).repeat(frames-image_from.shape[0], 1, 1, 1)), dim=0)

        mask = mask.unsqueeze(3).repeat(1, 1, 1, 3)

        if image_from.shape[1] != mask.shape[1] or image_from.shape[2] != mask.shape[2]:
            mask = comfy.utils.common_upscale(mask.permute([0,3,1,2]), image_from.shape[2], image_from.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])

        out = mask * image_to + (1 - mask) * image_from

        return (out, )

```
