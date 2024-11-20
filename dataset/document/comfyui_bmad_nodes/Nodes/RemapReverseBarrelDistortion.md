---
tags:
- ImageTransformation
---

# RemapReverseBarrelDistortion
## Documentation
- Class name: `RemapReverseBarrelDistortion`
- Category: `Bmad/CV/Transform`
- Output node: `False`

This node is designed to apply a reverse barrel distortion effect to images. It utilizes parameters to adjust the distortion effect, allowing for the correction of barrel distortion or the application of reverse effects based on the specified parameters.
## Input types
### Required
- **`a`**
    - Coefficient 'a' influences the primary distortion effect, playing a crucial role in the barrel distortion correction or application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`b`**
    - Coefficient 'b' serves to adjust the distortion effect further, complementing the primary coefficient 'a' in fine-tuning the distortion.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`c`**
    - Coefficient 'c' is used alongside 'a' and 'b' to refine the distortion effect, offering additional control over the distortion characteristics.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`use_inverse_variant`**
    - This boolean parameter determines whether to use an inverse variant of the distortion formula, affecting the overall distortion effect.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`d`**
    - An optional coefficient that can be used to extend the distortion effect, providing even finer control over the distortion characteristics.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The output is a remapped image with the reverse barrel distortion applied, based on the input parameters.
    - Python dtype: `Tuple[np.ndarray, np.ndarray, NoneType]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapReverseBarrelDistortion(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return RemapBarrelDistortion.BARREL_DIST_TYPES()

    def send_remap(self, a, b, c, use_inverse_variant, d=None):
        from .utils.remaps import remap_reverse_barrel_distortion
        return ({
                    "func": remap_reverse_barrel_distortion,
                    "xargs": [a, b, c, d, use_inverse_variant]
                },)

```
