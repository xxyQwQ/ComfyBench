---
tags:
- Mask
- MaskGeneration
---

# ChameleonMask
## Documentation
- Class name: `ChameleonMask`
- Category: `Bmad/CV/C.Photography`
- Output node: `False`

The ChameleonMask node is designed to adaptively blend or modify images or masks based on certain criteria or conditions. It aims to provide a flexible approach to image manipulation, allowing for dynamic adjustments that cater to the specific needs of the input data.
## Input types
### Required
- **`dst`**
    - Specifies the destination image over which the source image or mask will be applied, serving as the backdrop for the operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`src`**
    - Defines the source image or mask to be blended or modified onto the destination image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`thresh_blur`**
    - Sets the threshold for blurring, affecting the smoothness of the mask edges.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`close_dist`**
    - Determines the distance for the closing operation, impacting the closure of small holes in the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`open_dist`**
    - Specifies the distance for the opening operation, influencing the removal of small objects from the foreground of the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`size_dist`**
    - Controls the size distribution for filtering, affecting the scale of features retained in the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`mask_blur`**
    - Defines the blur level for the mask, impacting the overall softness and blending of edges.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`contrast_adjust`**
    - Adjusts the contrast of the image, enhancing or reducing the visual differences between elements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mode`**
    - Selects the operational mode, determining the specific blending or modification technique used.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`output_format`**
    - Chooses the output format of the image, affecting the visual style and compatibility.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`optional_roi_mask`**
    - An optional mask defining regions of interest, focusing the operation on specific areas of the image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after adaptive blending or modification, showcasing the applied effects.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ChameleonMask:  # wtf would I name this node as?
    mode_func_map = {
        "GRAY": lambda i: cv.cvtColor(i, cv.COLOR_BGR2GRAY),
        "VALUE": lambda i: cv.cvtColor(i, cv.COLOR_RGB2HSV)[:, :, 2],
        "LIGHTNESS": lambda i: cv.cvtColor(i, cv.COLOR_RGB2HLS)[:, :, 1],

        # not sure if these would be useful, but costs nothing to leave them here
        "HUE": lambda i: cv.cvtColor(i, cv.COLOR_RGB2HSV)[:, :, 0],
        "SATURATION (HSV)": lambda i: cv.cvtColor(i, cv.COLOR_RGB2HSV)[:, :, 1],
        "SATURATION (HSL)": lambda i: cv.cvtColor(i, cv.COLOR_RGB2HLS)[:, :, 2],
    }
    modes = list(mode_func_map.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dst": ("IMAGE",),
                "src": ("IMAGE",),
                "thresh_blur": ("INT", {"default": 30, "min": 2, "step": 2}),
                "close_dist": ("INT", {"default": 32, "min": 0, "step": 1}),
                "open_dist": ("INT", {"default": 32, "min": 0, "step": 1}),
                "size_dist": ("INT", {"default": 8, "min": -99999, "step": 1}),
                "mask_blur": ("INT", {"default": 64, "min": 0, "step": 2}),
                "contrast_adjust": ("FLOAT", {"default": 2.4, "min": 0, "max": 20, "step": .5}),
                "mode": (cls.modes, {"default": cls.modes[0]}),
                "output_format": (image_output_formats_options, {
                    "default": image_output_formats_options[0]
                }),

            },
            "optional": {
                "optional_roi_mask": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    FUNCTION = "create_mask"
    CATEGORY = f"{cv_category_path}/C.Photography"

    def create_mask(self, src, dst, thresh_blur, close_dist, open_dist, size_dist, mask_blur,
                    contrast_adjust, mode: str, output_format, optional_roi_mask=None):
        src = tensor2opencv(src)
        dst = tensor2opencv(dst)

        thresh_blur += 1
        if mask_blur > 0:
            mask_blur += 1

        # compute the difference between images based on mode
        src = self.mode_func_map[mode](src)  # type:ignore
        dst = self.mode_func_map[mode](dst)  # type:ignore
        diff = cv.absdiff(src, dst)

        if mode == "HUE":
            diff = np.minimum(diff, 180 - diff)

        # binary thresholding
        # _, mask = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)
        diff = cv.GaussianBlur(diff, (thresh_blur, thresh_blur), 0)
        _, mask = cv.threshold(diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        if optional_roi_mask is not None:
            optional_roi_mask = tensor2opencv(optional_roi_mask, 1)
            mask[optional_roi_mask < 127] = 0

        # morphological closing > closing > dilate/erode
        if close_dist > 0:
            close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_dist, close_dist))
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_kernel)
        if open_dist > 0:
            open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (open_dist, open_dist))
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, open_kernel)

        if size_dist > 0:
            size_op = cv.MORPH_DILATE
            size = size_dist
        else:
            size_op = cv.MORPH_ERODE
            size = abs(size_dist)
        if size_dist != 0:
            size_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))
            mask = cv.morphologyEx(mask, size_op, size_kernel)

        # gaussian blur + contrast adjust
        if mask_blur > 0:
            mask = cv.GaussianBlur(mask, (mask_blur, mask_blur), 0)
        mask = cv.convertScaleAbs(mask, alpha=1 + contrast_adjust, beta=0)  # / 100, beta=0)

        # convert to target format and output as tensor
        # note: diff is only meant to be used for debug purposes
        mask = maybe_convert_img(mask, 1, image_output_formats_options_map[output_format])
        diff = opencv2tensor(diff)
        mask = opencv2tensor(mask)

        return (mask, diff,)

```
