---
tags:
- Image
---

# MorphologicSkeletoning
## Documentation
- Class name: `MorphologicSkeletoning`
- Category: `Bmad/CV/Morphology`
- Output node: `False`

The MorphologicSkeletoning node is designed to perform skeletonization on binary images. It transforms the input image into its skeletal representation, reducing all objects to lines without changing the fundamental structure.
## Input types
### Required
- **`src`**
    - The source image to be skeletonized. It is crucial for the operation as it directly influences the skeletonization process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The skeletal representation of the input image, highlighting its essential structure with minimal thickness.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MorphologicSkeletoning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute"
    CATEGORY = f"{cv_category_path}/Morphology"

    def compute(self, src):
        from skimage.morphology import skeletonize
        img = tensor2opencv(src, 1)
        _, img = cv.threshold(img, 127, 1, cv.THRESH_BINARY)  # ensure it is binary and set max value to 1.
        skel = skeletonize(img) * 255
        img = opencv2tensor(skel)
        return (img,)

```
