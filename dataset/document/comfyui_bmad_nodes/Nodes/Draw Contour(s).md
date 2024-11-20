---
tags:
- Contour
- Image
---

# Draw Contour(s)
## Documentation
- Class name: `Draw Contour(s)`
- Category: `Bmad/CV/Contour`
- Output node: `False`

The DrawContours node is designed for visualizing contours on images by drawing them over the input image. It allows for selective contour visualization and customization of the contour appearance, such as color and thickness, enhancing the interpretability of contour-based analyses in computer vision tasks.
## Input types
### Required
- **`image`**
    - The input image on which contours will be drawn. It serves as the background for contour visualization.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`contours`**
    - A collection of contours to be drawn on the image. Each contour is defined by a series of points.
    - Comfy dtype: `CV_CONTOURS`
    - Python dtype: `List[List[torch.Tensor]]`
- **`index_to_draw`**
    - Specifies which contour from the contours list to draw. A value of -1 indicates that all contours should be drawn.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`thickness`**
    - The thickness of the contour lines. A negative value fills the contour.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`color`**
    - The color of the contour lines in RGB format.
    - Comfy dtype: `COLOR`
    - Python dtype: `Tuple[int, int, int]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image with the specified contours drawn over it.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DrawContours:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "contours": ("CV_CONTOURS",),
                "index_to_draw": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 1000,
                    "step": 1
                }),
                "thickness": ("INT", {
                    "default": 5,
                    "min": -1,
                    "max": 32,
                    "step": 1
                }),
                "color": ("COLOR",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"

    CATEGORY = f"{cv_category_path}/Contour"

    def draw(self, image, contours, index_to_draw, color, thickness):
        background = tensor2opencv(image)

        um_image = cv.UMat(background)
        cv.drawContours(um_image, contours, index_to_draw, ImageColor.getcolor(color, "RGB"), thickness)
        contour_image = um_image.get()

        image = opencv2tensor(contour_image)

        return (image,)

```
