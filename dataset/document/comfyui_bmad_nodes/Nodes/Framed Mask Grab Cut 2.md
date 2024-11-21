---
tags:
- Segmentation
---

# Framed Mask Grab Cut 2
## Documentation
- Class name: `Framed Mask Grab Cut 2`
- Category: `Bmad/CV/GrabCut`
- Output node: `False`

This node applies the GrabCut algorithm with an enhanced framing mechanism to segment the foreground from the background in images. It allows for more precise control over the segmentation process by incorporating options to ignore certain areas as probable background, thereby improving the accuracy of the mask generated.
## Input types
### Required
- **`image`**
    - The input image on which the GrabCut algorithm will be applied. This parameter is crucial for defining the area of interest for foreground segmentation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`thresh_maybe`**
    - Threshold values to identify probable foreground regions. This helps in refining the segmentation process by distinguishing between probable foreground and other areas.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`thresh_sure`**
    - Threshold values to identify definite foreground regions. This parameter is essential for accurately segmenting the foreground from the background.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`iterations`**
    - The number of iterations the GrabCut algorithm will run. More iterations can lead to more accurate segmentation but at the cost of increased computation time.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`margin`**
    - Defines the margin size around the frame to be considered as definite background, aiding in better isolation of the foreground object.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_option`**
    - Specifies the framing options to include or exclude certain parts of the image from being considered as background, offering flexibility in foreground segmentation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`binary_threshold`**
    - The threshold value used to differentiate between probable foreground and background based on the 'thresh_maybe' and 'thresh_sure' parameters.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`maybe_black_is_sure_background`**
    - A boolean parameter that, if true, treats pixels darker than the binary threshold as definite background, further refining the segmentation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`output_format`**
    - The desired output format of the result, which can influence how the segmented mask is utilized in subsequent processes or visualizations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The segmented foreground mask generated by the GrabCut algorithm, highlighting the areas identified as foreground.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FramedMaskGrabCut2:
    # TODO option to ignore probable background in sure_thresh

    frame_options = ['FULL_FRAME', 'IGNORE_BOTTOM', 'IGNORE_TOP', 'IGNORE_RIGHT', 'IGNORE_LEFT', 'IGNORE_HORIZONTAL'
        , 'IGNORE_VERTICAL']
    frame_options_values = {
        'FULL_FRAME': 0,
        'IGNORE_BOTTOM': 1,
        'IGNORE_TOP': 2,
        'IGNORE_RIGHT': 4,
        'IGNORE_LEFT': 8,
        'IGNORE_HORIZONTAL': 12,
        'IGNORE_VERTICAL': 3,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "thresh_maybe": ("IMAGE",),
                "thresh_sure": ("IMAGE",),
                "iterations": ("INT", {
                    "default": 25,
                    "min": 0,
                    "max": 200,
                    "step": 1
                }),
                "margin": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "frame_option": (cls.frame_options, {
                    "default": 'FULL_FRAME'
                }),
                # source thresh may not be only 0s and 1s, use this as a safeguard
                "binary_threshold": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 255,
                    "step": 1
                }),
                "maybe_black_is_sure_background": ("BOOLEAN", {"default": False}),
                "output_format": (image_output_formats_options, {
                    "default": image_output_formats_options[0]
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "grab_cut"

    CATEGORY = f"{cv_category_path}/GrabCut"

    def grab_cut(self, image, thresh_maybe, thresh_sure, iterations,
                 margin, frame_option, binary_threshold,
                 maybe_black_is_sure_background, output_format):
        image = tensor2opencv(image)

        thresh_maybe = tensor2opencv(thresh_maybe, 1)
        thresh_sure = tensor2opencv(thresh_sure, 1)

        fg_model = np.zeros((1, 65), dtype="float")
        bg_model = np.zeros((1, 65), dtype="float")
        mask = np.full(image.shape[:2], cv.GC_PR_BGD, dtype=np.uint8)  # probable background
        mask[thresh_maybe >= binary_threshold] = cv.GC_PR_FGD  # probable foreground
        mask[thresh_sure >= binary_threshold] = cv.GC_FGD  # foreground

        frame_option = self.frame_options_values[frame_option]
        include_bottom = not (frame_option & self.frame_options_values['IGNORE_BOTTOM'])
        include_top = not (frame_option & self.frame_options_values['IGNORE_TOP'])
        include_right = not (frame_option & self.frame_options_values['IGNORE_RIGHT'])
        include_left = not (frame_option & self.frame_options_values['IGNORE_LEFT'])

        if include_bottom:
            mask[-margin:, :] = cv.GC_BGD
        if include_top:
            mask[0:margin, :] = cv.GC_BGD
        if include_right:
            mask[:, -margin:] = cv.GC_BGD
        if include_left:
            mask[:, 0:margin] = cv.GC_BGD

        if maybe_black_is_sure_background:
            mask[thresh_maybe < binary_threshold] = cv.GC_BGD  # background

        mask, bg_model, fg_model = cv.grabCut(image, mask, None, bg_model, fg_model, iterCount=iterations,
                                              mode=cv.GC_INIT_WITH_MASK)

        # generate mask with "pixels" classified as background/foreground
        output_mask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1)
        output_mask = (output_mask * 255).astype("uint8")

        output_mask = maybe_convert_img(output_mask, 1, image_output_formats_options_map[output_format])
        image = opencv2tensor(output_mask)

        return (image,)

```