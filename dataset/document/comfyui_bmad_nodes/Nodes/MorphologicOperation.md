---
tags:
- Image
---

# MorphologicOperation
## Documentation
- Class name: `MorphologicOperation`
- Category: `Bmad/CV/Morphology`
- Output node: `False`

This node provides a comprehensive suite of morphological operations for image processing, including erode, dilate, open, close, gradient, top hat, and bottom hat transformations. It allows for the manipulation of the structure of images using different kernel shapes and sizes, enabling a wide range of effects from refining image features to extracting structural elements.
## Input types
### Required
- **`src`**
    - The source image to be processed. It serves as the primary input for applying morphological operations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`operation`**
    - Specifies the morphological operation to be applied to the source image. This determines the type of structural manipulation performed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`kernel_type`**
    - Defines the shape of the kernel used in the morphological operation, influencing the effect on the image's structure.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`kernel_size_x`**
    - The horizontal size of the kernel. It affects the area of the image influenced by the operation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`kernel_size_y`**
    - The vertical size of the kernel. It affects the area of the image influenced by the operation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`iterations`**
    - The number of times the morphological operation is applied. Increasing this value intensifies the effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The result of applying the specified morphological operation to the source image, potentially altering its structure.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MorphologicOperation:
    # I did not want to make this node, but alas, I found no suit w/ the top/black hat operation
    # so might as well make a generic node w/ all the operations
    # just return as BW and implement convert node

    operation_map = {
        "MORPH_ERODE": cv.MORPH_ERODE,
        "MORPH_DILATE": cv.MORPH_DILATE,
        "MORPH_OPEN": cv.MORPH_OPEN,
        "MORPH_CLOSE": cv.MORPH_CLOSE,
        "MORPH_GRADIENT": cv.MORPH_GRADIENT,
        "MORPH_TOPHAT": cv.MORPH_TOPHAT,
        "MORPH_BLACKHAT": cv.MORPH_BLACKHAT,
    }
    operations = list(operation_map.keys())

    kernel_types_map = {
        "MORPH_RECT": cv.MORPH_RECT,
        "MORPH_ELLIPSE": cv.MORPH_ELLIPSE,
        "MORPH_CROSS": cv.MORPH_CROSS,
    }
    kernel_types = list(kernel_types_map.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",),
                "operation": (cls.operations, {"default": cls.operations[0]}),
                "kernel_type": (cls.kernel_types, {"default": cls.kernel_types[0]}),
                "kernel_size_x": ("INT", {"default": 4, "min": 2, "step": 2}),
                "kernel_size_y": ("INT", {"default": 4, "min": 2, "step": 2}),
                "iterations": ("INT", {"default": 1, "step": 1}),
            },
            # TODO maybe add optional input for custom kernel
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = f"{cv_category_path}/Morphology"

    def apply(self, src, operation, kernel_type, kernel_size_x, kernel_size_y, iterations):
        img = tensor2opencv(src, 1)
        kernel = cv.getStructuringElement(self.kernel_types_map[kernel_type], (kernel_size_x + 1, kernel_size_y + 1))
        for i in range(iterations):
            img = cv.morphologyEx(img, self.operation_map[operation], kernel)
        return (opencv2tensor(img),)

```
