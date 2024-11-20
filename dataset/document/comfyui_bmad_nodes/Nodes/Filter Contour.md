---
tags:
- Contour
- Image
---

# Filter Contour
## Documentation
- Class name: `Filter Contour`
- Category: `Bmad/CV/Contour`
- Output node: `False`

The Filter Contour node is designed to process and filter contours based on a custom fitness function. It evaluates each contour in the context of an image and an auxiliary contour, allowing for dynamic selection and manipulation of contours according to specific criteria.
## Input types
### Required
- **`contours`**
    - A list of contours to be filtered. The fitness function evaluates each contour to determine its suitability.
    - Comfy dtype: `CV_CONTOURS`
    - Python dtype: `List[np.ndarray]`
- **`fitness`**
    - A custom lambda function that defines the criteria for filtering contours. It is applied to each contour to compute a fitness score.
    - Comfy dtype: `STRING`
    - Python dtype: `Callable[[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]], float]`
- **`select`**
    - A selection mode that determines how the filtered contours are returned. It influences the output based on the fitness scores of the contours.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`image`**
    - The image in which the contours are found. It provides context for evaluating the fitness of each contour.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Optional[np.ndarray]`
- **`aux_contour`**
    - An auxiliary contour that can be used as additional context in the fitness function for filtering contours.
    - Comfy dtype: `CV_CONTOUR`
    - Python dtype: `Optional[np.ndarray]`
## Output types
- **`cv_contour`**
    - Comfy dtype: `CV_CONTOUR`
    - The primary contour selected based on the fitness function.
    - Python dtype: `np.ndarray`
- **`cv_contours`**
    - Comfy dtype: `CV_CONTOURS`
    - The filtered contours selected based on the fitness function.
    - Python dtype: `List[np.ndarray]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FilterContour:
    @staticmethod
    def MODE(cnts, fit):
        sorted_list = sorted(cnts, key=fit)
        return [sorted_list[len(sorted_list) // 2]]

    return_modes_map = {
        "MAX": lambda cnts, fit: [sorted(cnts, key=fit)[-1]],
        "MIN": lambda cnts, fit: [sorted(cnts, key=fit)[0]],
        "MODE": MODE,
        "FILTER": lambda cnts, fit: list(filter(fit, cnts)),
    }
    return_modes = list(return_modes_map.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "contours": ("CV_CONTOURS",),
                "fitness": ("STRING", {"multiline": True, "default":
                    "# Contour Fitness Function\n"}),
                "select": (cls.return_modes, {"default": cls.return_modes[0]})
            },
            "optional": {
                "image": ("IMAGE",),
                "aux_contour": ("CV_CONTOUR",)
            }
        }

    RETURN_TYPES = ("CV_CONTOUR", "CV_CONTOURS")
    FUNCTION = "filter"
    CATEGORY = f"{cv_category_path}/Contour"

    def filter(self, contours, fitness, select, image=None, aux_contour=None):
        import math
        import cv2
        import numpy

        if len(contours) == 0:
            print("Contour list is empty")
            return ([[]], contours)

        # region prepare inputs
        if image is not None:
            image = tensor2opencv(image)

        fitness = prepare_text_for_eval(fitness)

        # endregion

        # region available functions
        # cv methods, but cache them
        @cache_with_ids(single=False)
        def boundingRect(cnt):
            return cv.boundingRect(cnt)

        @cache_with_ids(single=False)
        def contourArea(cnt):
            return cv.contourArea(cnt)

        @cache_with_ids(single=False)
        def arcLength(cnt):
            return cv.arcLength(cnt, True)

        @cache_with_ids(single=True)
        def minAreaRect(cnt):
            return cv.minAreaRect(cnt)

        @cache_with_ids(single=True)
        def minEnclosingCircle(cnt):
            return cv.minEnclosingCircle(cnt)

        @cache_with_ids(single=True)
        def fitEllipse(cnt):
            return cv.fitEllipse(cnt)

        @cache_with_ids(single=True)
        def convexHull(cnt):
            return cv.convexHull(cnt)

        # useful properties; adapted from multiple sources, including cv documentation
        @cache_with_ids(single=True)
        def aspect_ratio(cnt):
            _, _, w, h = boundingRect(cnt)
            return float(w) / h

        @cache_with_ids(single=True)
        def extent(cnt):
            area = contourArea(cnt)
            _, _, w, h = boundingRect(cnt)
            rect_area = w * h
            return float(area) / rect_area

        @cache_with_ids(single=True)
        def solidity(cnt):
            area = contourArea(cnt)
            hull = convexHull(cnt)
            hull_area = contourArea(hull)
            return float(area) / hull_area

        @cache_with_ids(single=True)
        def equi_diameter(cnt):
            area = contourArea(cnt)
            return math.sqrt(4 * area / math.pi)

        @cache_with_ids(single=True)
        def center(cnt):
            m = cv.moments(cnt)
            c_x = int(m["m10"] / m["m00"])
            c_y = int(m["m01"] / m["m00"])
            return c_x, c_y

        @cache_with_ids(single=False)
        def contour_mask(cnt, img):
            if len(img.shape) > 2:
                height, width, _ = img.shape
            else:
                height, width = img.shape

            mask = numpy.zeros((height, width, 1), numpy.uint8)
            cv.drawContours(mask, [cnt], 0, 255, -1)
            return mask

        @cache_with_ids(single=True)
        def mean_color(cnt, img):
            return cv.mean(img, mask=contour_mask(cnt, img))

        @cache_with_ids(single=True)
        def mean_intensity(cnt, img):
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            return mean_color(cnt, gray)[0]

        @cache_with_ids(single=True)
        def extreme_points(cnt):
            l = tuple(cnt[cnt[:, :, 0].argmin()][0])
            r = tuple(cnt[cnt[:, :, 0].argmax()][0])
            t = tuple(cnt[cnt[:, :, 1].argmin()][0])
            b = tuple(cnt[cnt[:, :, 1].argmax()][0])
            return {"top": t, "right": r, "bottom": b, "left": l}

        def intercepts_mask(cnt, img):  # where img should be a binary mask
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            intersection = cv2.bitwise_and(
                gray, cv2.drawContours(np.zeros_like(gray), [cnt], 0, 255, thickness=cv2.FILLED))
            return cv2.countNonZero(intersection) > 0

        # endregion

        available_funcs = {}
        for key, value in locals().items():
            if callable(value):
                available_funcs[key] = value

        fitness = eval(f"lambda c, i, a: {fitness}", {
            "__builtins__": {},
            "tuple": tuple, "list": list,
            'm': math, 'cv': cv2, 'np': numpy,
            **available_funcs
        }, {})

        ret = self.return_modes_map[select](contours, lambda c: fitness(c, image, aux_contour))
        return (ret[0], ret,)

```
