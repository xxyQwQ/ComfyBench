---
tags:
- BackgroundRemoval
- Image
---

# 🔧 Image Remove Background
## Documentation
- Class name: `ImageRemoveBackground+`
- Category: `essentials/image manipulation`
- Output node: `False`

This node is designed for removing the background from images using the rembg library. It processes each image in the input, applies the background removal algorithm, and returns the foreground image along with a mask indicating the areas of the image that were kept.
## Input types
### Required
- **`rembg_session`**
    - The session object for the rembg library, used to manage settings and configurations for the background removal process.
    - Comfy dtype: `REMBG_SESSION`
    - Python dtype: `rembg.Session`
- **`image`**
    - The input image or images to have their background removed. The node supports batch processing of multiple images.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The foreground of the input image(s) after background removal, with the background areas made transparent or removed.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - A binary mask indicating the areas of the image that were kept during the background removal process.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageRemoveBackground:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "rembg_session": ("REMBG_SESSION",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, rembg_session, image):
        from rembg import remove as rembg

        image = image.permute([0, 3, 1, 2])
        output = []
        for img in image:
            img = T.ToPILImage()(img)
            img = rembg(img, session=rembg_session)
            output.append(T.ToTensor()(img))

        output = torch.stack(output, dim=0)
        output = output.permute([0, 2, 3, 1])
        mask = output[:, :, :, 3] if output.shape[3] == 4 else torch.ones_like(output[:, :, :, 0])

        return(output[:, :, :, :3], mask,)

```
