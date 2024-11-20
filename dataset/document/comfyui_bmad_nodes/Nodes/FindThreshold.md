---
tags:
- Image
- ImageThresholding
---

# FindThreshold
## Documentation
- Class name: `FindThreshold`
- Category: `Bmad/CV/Thresholding`
- Output node: `False`

The FindThreshold node is designed to dynamically determine an optimal threshold value for image processing tasks. It systematically searches through a specified range of threshold values, applying each to the source image and evaluating the result based on a user-defined condition. This approach allows for adaptive thresholding, enabling the selection of a threshold that best meets the criteria for a given image or set of images.
## Input types
### Required
- **`src`**
    - The source image on which thresholding operations are to be performed. It serves as the primary input for the threshold search process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`start_at`**
    - Specifies the starting point of the threshold value range to be considered in the search process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at`**
    - Defines the end point of the threshold value range for the search, allowing the node to limit its evaluation to a specific range.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`thresh_type`**
    - Determines the type of thresholding to be applied during the search process, influencing how the threshold values are evaluated against the source image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`downscale_factor`**
    - A factor by which the source image is downscaled before thresholding, optimizing the search process by reducing computational load.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`condition`**
    - A user-defined condition (expressed as a string of Python code) that evaluates the effectiveness of each threshold value applied, guiding the selection of the optimal threshold.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after applying the optimal threshold found through the search process.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FindThreshold:
    """
    simple cond examples:
        cv.countNonZero(t)  > 100  # the number of non black pixels (white when using binary thresh type)
        (t.size - cv.countNonZero(t)) / t.size > .50 # the percentage of black pixels is higher than 50%
        # TODO can detect some shape(s) (requires optional inputs, and for current output maybe not that useful
    if none is found, returns the last
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",),
                "start_at": ("INT", {"default": 1, "min": 1, "max": 255, "step": 1}),
                "end_at": ("INT", {"default": 255, "min": 1, "max": 255, "step": 1}),
                "thresh_type": (thresh_types, {"default": thresh_types[0]}),
                "downscale_factor": ("INT", {"default": 2, "min": 1, "step": 1}),
                "condition": ("STRING", {"multiline": True, "default":
                    "# Some expression that returns True or False\n"}),
            },
        }  # TODO optional inputs

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "search"
    CATEGORY = f"{cv_category_path}/Thresholding"

    def search(self, src, start_at, end_at, thresh_type, downscale_factor, condition):
        import cv2
        import numpy
        import math

        o_img = tensor2opencv(src, 1)
        height, width = tuple(o_img.shape)
        img = cv.resize(o_img, (height // downscale_factor, width // downscale_factor), interpolation=cv.INTER_AREA)

        max_v = max(start_at, end_at)
        min_v = min(start_at, end_at)
        range_to_check = range(min_v, max_v + 1)
        if end_at < start_at:
            range_to_check = range_to_check.__reversed__()

        condition = prepare_text_for_eval(condition)
        cond_check = eval(f"lambda t: {condition}", {
            "__builtins__": {},
            "tuple": tuple, "list": list,
            'm': math, 'cv': cv2, 'np': numpy
        }, {})

        thresh_value = end_at
        for i in range_to_check:
            _, thresh = cv.threshold(img, i, 255, thresh_types_map[thresh_type])
            if cond_check(thresh):
                thresh_value = i
                break

        _, img = cv.threshold(o_img, thresh_value, 255, thresh_types_map[thresh_type])
        img = opencv2tensor(img)
        return (img,)

```
