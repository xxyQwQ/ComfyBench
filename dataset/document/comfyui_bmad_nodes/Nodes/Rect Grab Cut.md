---
tags:
- Segmentation
---

# Rect Grab Cut
## Documentation
- Class name: `Rect Grab Cut`
- Category: `Bmad/CV/GrabCut`
- Output node: `False`

The RectGrabCut node is designed for image segmentation using the GrabCut algorithm with a predefined rectangular area. It aims to separate the foreground from the background within the specified rectangle, enhancing image processing tasks by allowing for more precise object isolation.
## Input types
### Required
- **`image`**
    - The input image to be segmented. It plays a crucial role in determining the effectiveness of the foreground-background separation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`x1`**
    - The x-coordinate of the top left corner of the rectangle.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y1`**
    - The y-coordinate of the top left corner of the rectangle.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`x2`**
    - The x-coordinate of the bottom right corner of the rectangle.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y2`**
    - The y-coordinate of the bottom right corner of the rectangle.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`iterations`**
    - Specifies the number of iterations the GrabCut algorithm should run, affecting the accuracy and detail of the segmentation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`output_format`**
    - Defines the format of the output image, influencing how the segmented image is presented and utilized in subsequent processes.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The segmented image with the foreground isolated from the background, ready for further processing or analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RectGrabCut:
    # TODO add option to crop or just leave as 0 the section outside the rect
    # TODO maybe add option to exclude PR_BGD or include PR_FGD in outputMask

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x1": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 2000,
                    "step": 1
                }),
                "y1": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 2000,
                    "step": 1
                }),
                "x2": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 2000,
                    "step": 1
                }),
                "y2": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 2000,
                    "step": 1
                }),
                "iterations": ("INT", {
                    "default": 25,
                    "min": 0,
                    "max": 200,
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

    def grab_cut(self, image, iterations, x1, y1, x2, y2, output_format):
        image = tensor2opencv(image)

        fg_model = np.zeros((1, 65), dtype="float")
        bg_model = np.zeros((1, 65), dtype="float")
        mask = np.zeros(image.shape[:2], dtype="uint8")
        rect = (x1, y1, x2, y2)

        mask, bg_model, fg_model = cv.grabCut(image, mask, rect, bg_model,
                                              fg_model, iterCount=iterations, mode=cv.GC_INIT_WITH_RECT)

        # generate mask with "pixels" classified as background/foreground
        output_mask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),
                               0, 1)
        output_mask = (output_mask * 255).astype("uint8")

        output_mask = maybe_convert_img(output_mask, 1, image_output_formats_options_map[output_format])
        image = opencv2tensor(output_mask)
        # image = image[y1:y2, x1:x2] #TODO maybe add option whether to crop or not

        return (image,)

```
