---
tags:
- Color
- ImageEnhancement
---

# 🔧 Image Apply LUT
## Documentation
- Class name: `ImageApplyLUT+`
- Category: `essentials/image processing`
- Output node: `False`

This node applies a Look-Up Table (LUT) to an image or a batch of images to modify their appearance based on the LUT's defined transformations. It supports optional gamma correction, clipping of LUT values, and blending the transformed image with the original based on a specified strength. This process allows for sophisticated color grading and image effect applications.
## Input types
### Required
- **`image`**
    - The image or batch of images to which the LUT will be applied. This is the primary input that undergoes transformation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`lut_file`**
    - The filename of the Look-Up Table (LUT) to be applied. This determines the specific color transformation to be executed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`gamma_correction`**
    - A boolean flag indicating whether gamma correction should be applied to the image before and after LUT application, enhancing the visual quality.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`clip_values`**
    - A boolean flag that, when true, clips the LUT values to ensure they stay within the defined domain, preventing potential color distortion.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`strength`**
    - A float value between 0 and 1 indicating the blend strength of the LUT-applied image with the original image, allowing for subtle to full application of the LUT effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The transformed image or batch of images after the LUT has been applied, optionally gamma-corrected and blended with the original based on the specified strength.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageApplyLUT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_file": ([f for f in os.listdir(LUTS_DIR) if f.lower().endswith('.cube')], ),
                "gamma_correction": ("BOOLEAN", { "default": True }),
                "clip_values": ("BOOLEAN", { "default": True }),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1 }),
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    # TODO: check if we can do without numpy
    def execute(self, image, lut_file, gamma_correction, clip_values, strength):
        from colour.io.luts.iridas_cube import read_LUT_IridasCube

        device = image.device
        lut = read_LUT_IridasCube(os.path.join(LUTS_DIR, lut_file))
        lut.name = lut_file

        if clip_values:
            if lut.domain[0].max() == lut.domain[0].min() and lut.domain[1].max() == lut.domain[1].min():
                lut.table = np.clip(lut.table, lut.domain[0, 0], lut.domain[1, 0])
            else:
                if len(lut.table.shape) == 2:  # 3x1D
                    for dim in range(3):
                        lut.table[:, dim] = np.clip(lut.table[:, dim], lut.domain[0, dim], lut.domain[1, dim])
                else:  # 3D
                    for dim in range(3):
                        lut.table[:, :, :, dim] = np.clip(lut.table[:, :, :, dim], lut.domain[0, dim], lut.domain[1, dim])

        out = []
        for img in image: # TODO: is this more resource efficient? should we use a batch instead?
            lut_img = img.cpu().numpy().copy()

            is_non_default_domain = not np.array_equal(lut.domain, np.array([[0., 0., 0.], [1., 1., 1.]]))
            dom_scale = None
            if is_non_default_domain:
                dom_scale = lut.domain[1] - lut.domain[0]
                lut_img = lut_img * dom_scale + lut.domain[0]
            if gamma_correction:
                lut_img = lut_img ** (1/2.2)
            lut_img = lut.apply(lut_img)
            if gamma_correction:
                lut_img = lut_img ** (2.2)
            if is_non_default_domain:
                lut_img = (lut_img - lut.domain[0]) / dom_scale

            lut_img = torch.from_numpy(lut_img).to(device)
            if strength < 1.0:
                lut_img = strength * lut_img + (1 - strength) * img
            out.append(lut_img)

        out = torch.stack(out)

        return (out, )

```
