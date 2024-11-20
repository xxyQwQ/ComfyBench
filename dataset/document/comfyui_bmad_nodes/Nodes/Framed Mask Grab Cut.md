---
tags:
- Segmentation
---

# Framed Mask Grab Cut
## Documentation
- Class name: `Framed Mask Grab Cut`
- Category: `Bmad/CV/GrabCut`
- Output node: `False`

The Framed Mask Grab Cut node is designed to apply the GrabCut algorithm with additional framing options to segment the foreground from the background in images. It allows for the specification of probable and sure foreground and background areas, and incorporates frame options to exclude certain margins from consideration, enhancing the flexibility and accuracy of the segmentation process.
## Input types
### Required
- **`image`**
    - The input image on which the GrabCut algorithm will be applied. It serves as the primary data for foreground-background segmentation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`thresh`**
    - A thresholded image indicating areas that might be the foreground or background. This parameter helps in refining the segmentation by marking uncertain areas.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`iterations`**
    - The number of iterations the GrabCut algorithm should run, affecting the refinement of the segmentation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`margin`**
    - Specifies the margin size to be excluded from the frame options, allowing for more precise control over the area considered for segmentation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_option`**
    - Defines the frame options to apply, such as ignoring certain margins, to tailor the segmentation process to specific requirements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `int`
- **`threshold_FGD`**
    - The threshold value used to classify pixels in 'thresh' image as foreground.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`threshold_PR_FGD`**
    - The threshold value used to classify pixels in 'thresh' image as probable foreground.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`output_format`**
    - The desired format for the output image, allowing for compatibility with different processing or display requirements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The segmented image with the foreground isolated from the background, based on the GrabCut algorithm and additional framing options.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FramedMaskGrabCut:
    frame_options_values = {
        'FULL_FRAME': 0,
        'IGNORE_BOTTOM': 1,
        'IGNORE_TOP': 2,
        'IGNORE_RIGHT': 4,
        'IGNORE_LEFT': 8,
        'IGNORE_HORIZONTAL': 12,
        'IGNORE_VERTICAL': 3,
    }
    frame_options = list(frame_options_values.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "thresh": ("IMAGE",),
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
                    "default": cls.frame_options[0]
                }),

                # to only use PR FGD set threshold_FGD to 0
                # to only use only FGD set threshold_FGD to a lower value than threshold_PR_FGD
                # using one of these also works as a safeguard in case thresh has other values besides 0s and 1s
                "threshold_FGD": ("INT", {
                    "default": 250,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "threshold_PR_FGD": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 255,
                    "step": 1
                }),
                "output_format": (image_output_formats_options, {
                    "default": image_output_formats_options[0]
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "grab_cut"
    CATEGORY = f"{cv_category_path}/GrabCut"

    def grab_cut(self, image, thresh, iterations, margin, frame_option, threshold_FGD, threshold_PR_FGD, output_format):
        image = tensor2opencv(image)
        thresh = tensor2opencv(thresh, 1)

        assert image.shape[:2] == thresh.shape

        fg_model = np.zeros((1, 65), dtype="float")
        bg_model = np.zeros((1, 65), dtype="float")
        mask = np.full(image.shape[:2], cv.GC_PR_BGD, dtype=np.uint8)  # probable background
        # foreground and probable foreground
        if threshold_FGD > threshold_PR_FGD:
            mask[thresh >= threshold_PR_FGD] = cv.GC_PR_FGD
        if threshold_FGD > 0:
            mask[thresh >= threshold_FGD] = cv.GC_FGD

        # check what borders should be painted
        frame_option = self.frame_options_values[frame_option]
        include_bottom = not (frame_option & self.frame_options_values['IGNORE_BOTTOM'])
        include_top = not (frame_option & self.frame_options_values['IGNORE_TOP'])
        include_right = not (frame_option & self.frame_options_values['IGNORE_RIGHT'])
        include_left = not (frame_option & self.frame_options_values['IGNORE_LEFT'])

        # paint the borders as being background
        if include_bottom:
            mask[-margin:, :] = cv.GC_BGD
        if include_top:
            mask[0:margin, :] = cv.GC_BGD
        if include_right:
            mask[:, -margin:] = cv.GC_BGD
        if include_left:
            mask[:, 0:margin] = cv.GC_BGD

        mask, bg_model, fg_model = cv.grabCut(image, mask, None, bg_model, fg_model, iterCount=iterations,
                                              mode=cv.GC_INIT_WITH_MASK)

        # generate mask with "pixels" classified as background/foreground
        output_mask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1)

        output_mask = (output_mask * 255).astype("uint8")

        output_mask = maybe_convert_img(output_mask, 1, image_output_formats_options_map[output_format])
        image = opencv2tensor(output_mask)

        return (image,)

```
