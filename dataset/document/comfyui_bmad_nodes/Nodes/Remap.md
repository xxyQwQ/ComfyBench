---
tags:
- ImageTransformation
---

# Remap
## Documentation
- Class name: `Remap`
- Category: `Bmad/CV/Transform`
- Output node: `False`

The `Remap` node serves as a foundational class for various image remapping operations, providing a base structure and common functionalities for specialized remapping transformations. It abstracts the core mechanisms needed to perform complex image geometry alterations, enabling derived classes to implement specific remapping effects such as distortion correction, perspective adjustments, and custom image warping techniques.
## Input types
### Required
- **`remap`**
    - Specifies the remapping function to be applied, which is a core component of the remapping process, determining how pixels from the source image are mapped to the destination image.
    - Comfy dtype: `REMAP`
    - Python dtype: `function`
- **`src`**
    - The source image to be remapped, serving as the input for the remapping operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`interpolation`**
    - Defines the interpolation method used in the remapping process, affecting the quality and appearance of the output image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `int`
### Optional
- **`src_mask`**
    - An optional mask that specifies which parts of the source image should be considered for remapping.
    - Comfy dtype: `MASK`
    - Python dtype: `numpy.ndarray`
- **`output_with_alpha`**
    - A boolean flag indicating whether the output image should include an alpha channel, allowing for transparency effects.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The result of the remapping operation, which is a transformed version of the input image according to the specified remapping parameters.
    - Python dtype: `numpy.ndarray`
- **`mask`**
    - Comfy dtype: `MASK`
    - An optional output mask that corresponds to the remapped image, indicating areas of interest or exclusion.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class Remap:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "remap": ("REMAP", {"forceInput": True}),
            "src": ("IMAGE",),
            "interpolation": (interpolation_types, {"default": interpolation_types[2]}),
        },
            "optional": {
                "src_mask": ("MASK",),
                "output_with_alpha": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "transform"
    CATEGORY = f"{cv_category_path}/Transform"

    def transform(self, src, remap, interpolation, src_mask=None, output_with_alpha=False):
        src = tensor2opencv(src)
        dst_dims = remap["dims"] if "dims" in remap else src.shape[:2]
        func = remap["func"]
        xargs = remap["xargs"]
        # if src_mask is not defined set it to a blank canvas; otherwise, just unwrap it
        src_mask = np.ones(src.shape[:2]) * 255 if src_mask is None else tensor2opencv(src_mask, 1)

        if "custom" not in remap.keys():
            # generic application, using cv.remap
            xs, ys, bb = func(src, *xargs)
            remap_img = cv.remap(src, xs, ys, interpolation_types_map[interpolation])
            mask = cv.remap(src_mask, xs, ys, interpolation_types_map[interpolation])
        else:
            # non-generic application; replaces cv.remap w/ some other function.
            # so far only for user provided homography,
            #  to avoid a separate node, since technically it is also a remap and also uses the interpolation argument.
            custom_data = func(src, *xargs)
            remap_img, mask, bb = remap["custom"](custom_data, src, interpolation_types_map[interpolation], src_mask)

        if bb is not None:
            new_img = np.zeros((*dst_dims, 3))  # hope width and height are not swapped
            new_img[bb[1]:bb[3], bb[0]:bb[2], :] = remap_img
            remap_img = new_img
            new_img = np.zeros(dst_dims)  # was working previously without the batch dim; unsure if really needed
            new_img[bb[1]:bb[3], bb[0]:bb[2]] = mask
            mask = new_img

        if output_with_alpha:
            new_img = np.zeros((*dst_dims, 4))
            new_img[:, :, 0:3] = remap_img[:, :, :]
            new_img[:, :, 3] = mask[:, :]
            remap_img = new_img

        return (opencv2tensor(remap_img), opencv2tensor(mask))

```
