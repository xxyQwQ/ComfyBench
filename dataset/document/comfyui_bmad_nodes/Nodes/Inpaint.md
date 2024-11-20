---
tags:
- DepthMap
- Image
- Inpaint
---

# Inpaint
## Documentation
- Class name: `Inpaint`
- Category: `Bmad/CV/C.Photography`
- Output node: `False`

The Inpaint node is designed to reconstruct the missing or damaged parts of images using a specified inpainting technique. It leverages computer vision algorithms to fill in gaps or remove unwanted objects from images, based on the surrounding pixel information.
## Input types
### Required
- **`img`**
    - The image to be inpainted. It serves as the primary input where missing parts or imperfections are identified and subsequently reconstructed.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - A mask image indicating the areas to be inpainted. Pixels marked in the mask are the targets for inpainting, guiding the algorithm on where to apply the reconstruction process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`radius`**
    - Defines the neighborhood size around each point to consider for inpainting. A larger radius uses more surrounding pixels for reconstruction, potentially leading to smoother results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`flag`**
    - Specifies the inpainting technique to use. Different methods can produce varying results based on the algorithm's approach to filling in missing data.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The inpainted image, where the specified areas have been reconstructed using the chosen inpainting technique.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Inpaint:
    inpaint_method_map = {
        "TELEA": cv.INPAINT_TELEA,
        "NS": cv.INPAINT_NS,
    }
    inpaint_methods = list(inpaint_method_map.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img": ("IMAGE",),
                "mask": ("IMAGE",),
                "radius": ("INT", {"default": 3, "min": 0, "step": 1}),
                "flag": (cls.inpaint_methods, {"default": cls.inpaint_methods[0]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paint"
    CATEGORY = f"{cv_category_path}/C.Photography"

    def paint(self, img, mask, radius, flag):
        img = tensor2opencv(img)
        mask = tensor2opencv(mask, 1)
        dst = cv.inpaint(img, mask, radius, self.inpaint_method_map[flag])
        result = opencv2tensor(dst)
        return (result,)

```
