---
tags:
- Image
- ImageThresholding
---

# DistanceTransform
## Documentation
- Class name: `DistanceTransform`
- Category: `Bmad/CV/Thresholding`
- Output node: `False`

The DistanceTransform node applies a distance transform to a binary image, converting it into a grayscale image where each pixel's intensity represents its distance to the nearest background pixel. It supports various distance types and mask sizes, allowing for customization of the distance calculation.
## Input types
### Required
- **`binary_image`**
    - The binary image to which the distance transform will be applied. It is crucial for identifying foreground and background pixels for distance calculations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`distance_type`**
    - Specifies the type of distance calculation to use, such as Euclidean (L2), Manhattan (L1), or Chebyshev (C). It affects the way distances are computed in the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`mask_size`**
    - Determines the size of the mask used in the distance calculation, affecting the precision and computational cost of the transform.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output grayscale image where pixel intensities represent distances to the nearest background pixel.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DistanceTransform:
    distance_types_map = {
        "DIST_L2": cv.DIST_L2,
        "DIST_L1": cv.DIST_L1,
        "DIST_C": cv.DIST_C,
    }
    distance_types = list(distance_types_map.keys())

    mask_sizes_map = {
        "DIST_MASK_3": cv.DIST_MASK_3,
        "DIST_MASK_5": cv.DIST_MASK_5,
        "DIST_MASK_PRECISE": cv.DIST_MASK_PRECISE
    }
    mask_sizes = list(mask_sizes_map.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_image": ("IMAGE",),
                "distance_type": (cls.distance_types, {"default": cls.distance_types[0]}),
                "mask_size": (cls.mask_sizes, {"default": cls.mask_sizes[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = f"{cv_category_path}/Thresholding"

    def apply(self, binary_image, distance_type, mask_size):
        binary_image = tensor2opencv(binary_image, 1)
        distance_transform = cv.distanceTransform(
            binary_image,
            self.distance_types_map[distance_type],
            self.mask_sizes_map[mask_size])
        distance_transform = opencv2tensor(distance_transform)
        return (distance_transform,)

```
