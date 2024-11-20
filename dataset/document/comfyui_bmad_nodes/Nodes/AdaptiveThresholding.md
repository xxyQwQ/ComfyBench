---
tags:
- Image
- ImageThresholding
---

# AdaptiveThresholding
## Documentation
- Class name: `AdaptiveThresholding`
- Category: `Bmad/CV/Thresholding`
- Output node: `False`

The AdaptiveThresholding node applies adaptive thresholding techniques to images, dynamically adjusting the threshold value over different regions of the image based on local image characteristics. This method is particularly useful for images with varying lighting conditions, enhancing the ability to distinguish foreground from background.
## Input types
### Required
- **`src`**
    - The source image to be thresholded. It is crucial for defining the input on which the adaptive thresholding operation will be performed.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`max_value`**
    - The maximum intensity value that a pixel can have after thresholding. It determines the brightness of the foreground region in the output image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`adaptive_method`**
    - Specifies the algorithm to use for calculating the threshold for a particular region of the image. It affects how the threshold values are computed across the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`threshold_type`**
    - Determines the type of thresholding to be applied, influencing how pixels are classified into foreground and background.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`block_size`**
    - The size of the neighborhood area used to calculate the threshold for each pixel. It influences the granularity of the thresholding.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`c`**
    - A constant subtracted from the mean or weighted mean calculated. It allows fine-tuning of the thresholding process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The result of applying adaptive thresholding to the input image, enhancing the distinction between regions of different intensities.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class AdaptiveThresholding:
    adaptive_modes_map = {
        "ADAPTIVE_THRESH_MEAN_C": cv.ADAPTIVE_THRESH_MEAN_C,
        "ADAPTIVE_THRESH_GAUSSIAN_C": cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    }
    adaptive_modes = list(adaptive_modes_map.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",),
                "max_value": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                # maybe should just allow for 255? may just confuse some people that don't read documentation
                "adaptive_method": (cls.adaptive_modes, {"default": cls.adaptive_modes[1]}),
                "threshold_type": (thresh_types, {"default": thresh_types[0]}),
                "block_size": ("INT", {"default": 4, "min": 2, "step": 2}),
                "c": ("INT", {"default": 2, "min": -999, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "thresh"
    CATEGORY = f"{cv_category_path}/Thresholding"

    def thresh(self, src, max_value, adaptive_method, threshold_type, block_size, c):
        # maybe allow to use from a specific channel 1st? nah, just create a node to fetch the channel
        # might be useful for other nodes
        src = tensor2opencv(src, 1)
        src = cv.adaptiveThreshold(src, max_value, self.adaptive_modes_map[adaptive_method],
                                   thresh_types_map[threshold_type], block_size + 1, c)
        src = cv.cvtColor(src, cv.COLOR_GRAY2RGB)
        src = opencv2tensor(src)
        return (src,)

```
