---
tags:
- ImageTransformation
---

# RemapBarrelDistortion
## Documentation
- Class name: `RemapBarrelDistortion`
- Category: `Bmad/CV/Transform`
- Output node: `False`

This node is designed to apply barrel distortion to images, a common technique for correcting or introducing distortion based on lens characteristics. It can undistort images from camera and lens combinations or create specific types of distortion such as barrel, pincushion, or mustache distortion.
## Input types
### Required
- **`a`**
    - Coefficient 'a' influences the primary distortion effect on the image, playing a crucial role in the barrel distortion calculation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`b`**
    - Coefficient 'b' serves as a secondary factor in the distortion effect, modifying the curvature introduced by the primary coefficient.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`c`**
    - Coefficient 'c' acts as a tertiary modifier for the distortion effect, fine-tuning the overall shape of the distortion.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`use_inverse_variant`**
    - This boolean flag determines whether an alternative formula is used for computing the distortion, allowing for different distortion effects.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`d`**
    - An optional coefficient that can further adjust the distortion effect when provided.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The output is the result of applying the specified barrel distortion to the input image, which may include corrections or intentional distortions based on the provided coefficients.
    - Python dtype: `Tuple[ndarray, ndarray, NoneType]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapBarrelDistortion(RemapBase):
    @staticmethod
    def BARREL_DIST_TYPES():
        return {
            "required":
                {
                    "a": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.00001}),
                    "b": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.00001}),
                    "c": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.00001}),
                    "use_inverse_variant": ("BOOLEAN", {"default": True})
                },
            "optional": {
                "d": ("FLOAT", {"forceInput": True})
            }
        }

    @classmethod
    def INPUT_TYPES(cls):
        return RemapBarrelDistortion.BARREL_DIST_TYPES()
        # inputs = RemapBarrelDistortion.BARREL_DIST_F_TYPES()
        # inputs["required"]["use_inverse_variant"] = ("BOOLEAN", {"default": True})
        # return inputs

    def send_remap(self, a, b, c, use_inverse_variant, d=None):
        from .utils.remaps import remap_barrel_distortion
        return ({
                    "func": remap_barrel_distortion,
                    "xargs": [a, b, c, d, use_inverse_variant]
                },)

```
