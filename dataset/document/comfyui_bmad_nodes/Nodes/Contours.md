---
tags:
- Contour
- Image
---

# Contours
## Documentation
- Class name: `Contours`
- Category: `Bmad/CV/Contour`
- Output node: `False`

The Contours node is designed to identify and extract contours from an image based on specified retrieval and approximation modes. It converts the input image to grayscale, applies thresholding if necessary, and utilizes OpenCV's findContours method to detect contours, providing a foundational step for further image analysis or manipulation.
## Input types
### Required
- **`image`**
    - The input image tensor that contours will be extracted from. It's crucial as the source for contour detection.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`retrieval_mode`**
    - Specifies the contour retrieval mode, affecting how contours are organized and retrieved. It plays a key role in defining the hierarchy of the contours.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`approximation_mode`**
    - Determines the method used to approximate the contours. This affects the level of detail of the extracted contours.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`cv_contours`**
    - Comfy dtype: `CV_CONTOURS`
    - The detected contours as a list of points.
    - Python dtype: `List[List[torch.Tensor]]`
- **`cv_contour`**
    - Comfy dtype: `CV_CONTOUR`
    - A single contour selected from the detected contours, if applicable.
    - Python dtype: `List[torch.Tensor]`
- **`cv_contours_hierarchy`**
    - Comfy dtype: `CV_CONTOURS_HIERARCHY`
    - The hierarchical information of the contours, providing details on the contour structure and relationships.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Contours:
    """
    Note:
    The image is converted to grey, but no threshold is applied.
    Apply the thresholding before using and feed a black and white image.
    """

    approximation_modes_map = {
        'CHAIN_APPROX_NONE': cv.CHAIN_APPROX_NONE,
        'CHAIN_APPROX_SIMPLE': cv.CHAIN_APPROX_SIMPLE,
        'CHAIN_APPROX_TC89_L1': cv.CHAIN_APPROX_TC89_L1,
        'CHAIN_APPROX_TC89_KCOS': cv.CHAIN_APPROX_TC89_KCOS
    }
    approximation_modes = list(approximation_modes_map.keys())

    retrieval_modes_map = {
        'RETR_EXTERNAL': cv.RETR_EXTERNAL,
        'RETR_LIST': cv.RETR_LIST,
        'RETR_CCOMP': cv.RETR_CCOMP,
        'RETR_TREE': cv.RETR_TREE,
        'RETR_FLOODFILL': cv.RETR_FLOODFILL
    }
    retrieval_modes = list(retrieval_modes_map.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "retrieval_mode": (cls.retrieval_modes, {"default": "RETR_LIST"}),
                "approximation_mode": (cls.approximation_modes, {"default": "CHAIN_APPROX_SIMPLE"}),
            },
        }

    RETURN_TYPES = ("CV_CONTOURS", "CV_CONTOUR", "CV_CONTOURS_HIERARCHY")
    FUNCTION = "find_contours"
    CATEGORY = f"{cv_category_path}/Contour"
    OUTPUT_IS_LIST = (False, True, False)

    def find_contours(self, image, retrieval_mode, approximation_mode):
        image = tensor2opencv(image)
        thresh = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        # no thresh applied here, non zeroes are treated as 1 according to documentation;
        # thresh should have been already applied to the image, before passing it to this node.

        contours, hierarchy = cv.findContours(
            thresh,
            self.retrieval_modes_map[retrieval_mode],
            self.approximation_modes_map[approximation_mode])

        return (contours, contours, hierarchy,)

```
