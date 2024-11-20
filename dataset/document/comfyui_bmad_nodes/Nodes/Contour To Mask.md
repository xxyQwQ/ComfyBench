---
tags:
- Contour
- Image
---

# Contour To Mask
## Documentation
- Class name: `Contour To Mask`
- Category: `Bmad/CV/Contour`
- Output node: `False`

The Contour To Mask node transforms a given contour into a mask on the specified image, allowing for the visualization or application of contours in image processing tasks. It supports customization of the output image format.
## Input types
### Required
- **`image`**
    - The input image on which the contour will be drawn to create a mask. It serves as the canvas for the contour transformation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`contour`**
    - The contour to be transformed into a mask on the image. This parameter defines the shape and location of the mask.
    - Comfy dtype: `CV_CONTOUR`
    - Python dtype: `CV_CONTOUR`
- **`output_format`**
    - Specifies the desired output image format, allowing for flexibility in how the resulting masked image is encoded or saved.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `image_output_formats_options`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is an image with the specified contour transformed into a mask, reflecting the shape and position of the contour on the original image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ContourToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "contour": ("CV_CONTOUR",),
                "output_format": (image_output_formats_options, {
                    "default": image_output_formats_options[0]
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = f"{cv_category_path}/Contour"

    def draw(self, image, contour, output_format):
        image = tensor2opencv(image, 1)
        image = np.zeros(image.shape, dtype=np.uint8)
        cv.drawContours(image, [contour], 0, (255), -1)
        image = maybe_convert_img(image, 1, image_output_formats_options_map[output_format])
        image = opencv2tensor(image)
        return (image,)

```
