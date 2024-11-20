---
tags:
- Image
- ImageThresholding
---

# OtsuThreshold
## Documentation
- Class name: `OtsuThreshold`
- Category: `Bmad/CV/Thresholding`
- Output node: `False`

The OtsuThreshold node applies Otsu's thresholding method to an image to separate the foreground from the background. It optionally preprocesses the image with Gaussian blur to reduce noise and improve the thresholding result.
## Input types
### Required
- **`image`**
    - The input image to be thresholded. Otsu's method is applied to this image to determine the optimal threshold value for separating foreground and background.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`threshold_type`**
    - Specifies the type of thresholding to apply in conjunction with Otsu's method, allowing for customization of the thresholding process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Enum`
- **`gaussian_blur_x`**
    - The kernel width for the Gaussian blur applied to the image before thresholding. A larger value reduces noise but can blur edges.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`gaussian_blur_y`**
    - The kernel height for the Gaussian blur applied to the image before thresholding. Works in conjunction with gaussian_blur_x to define the blur extent.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`gaussian_border_type`**
    - Defines the border type used in the Gaussian blur process, affecting how image edges are handled during blurring.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Enum`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after applying Otsu's thresholding. The image is segmented into foreground and background, with noise reduced if Gaussian blur was applied.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class OtsuThreshold:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # "channel": (s.channels, {"default": "greyscale"}),
                "threshold_type": (thresh_types, {"default": thresh_types[0]}),
                "gaussian_blur_x": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 200,
                    "step": 2
                }),
                "gaussian_blur_y": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 200,
                    "step": 2
                }),
                "gaussian_border_type": (border_types, {"default": border_types[0]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "otsu_thresthold"
    CATEGORY = f"{cv_category_path}/Thresholding"

    def otsu_thresthold(self, image, threshold_type, gaussian_blur_x, gaussian_blur_y, gaussian_border_type):
        image = tensor2opencv(image, 1)
        if gaussian_blur_x > 0 and gaussian_blur_y > 0:
            image = cv.GaussianBlur(image, (gaussian_blur_x + 1, gaussian_blur_y + 1),
                                    border_types_map[gaussian_border_type])
        _, image = cv.threshold(image, 0, 255, thresh_types_map[threshold_type] + cv.THRESH_OTSU)
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        image = opencv2tensor(image)
        return (image,)

```
