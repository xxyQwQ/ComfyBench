---
tags:
- Color
---

# KMeansColor
## Documentation
- Class name: `KMeansColor`
- Category: `Bmad/CV/Color A.`
- Output node: `False`

This node applies the K-means clustering algorithm to an image to identify and extract a specified number of dominant colors. It leverages OpenCV's implementation of K-means for efficient color quantization, allowing for the simplification of the image's color palette.
## Input types
### Required
- **`image`**
    - The input image to be processed for color quantization. It serves as the basis for identifying dominant colors through clustering.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`number_of_colors`**
    - Specifies the number of dominant colors to identify in the image. It determines the granularity of the color quantization.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`max_iterations`**
    - The maximum number of iterations for the K-means algorithm to run, ensuring the process terminates.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`eps`**
    - The epsilon value for convergence criteria in K-means, dictating the algorithm's precision and stopping condition.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image with its colors quantized to the specified dominant colors.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class KMeansColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "number_of_colors": ("INT", {"default": 2, "min": 1}),
            "max_iterations": ("INT", {"default": 100}),
            "eps": ("FLOAT", {"default": .2, "step": 0.05})
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_colors"
    CATEGORY = f"{cv_category_path}/Color A."

    def get_colors(self, image, number_of_colors, max_iterations, eps):
        image = tensor2opencv(image, 3)
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)

        # define criteria and apply kmeans
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iterations, eps)
        _, labels, centers = cv.kmeans(pixels, number_of_colors, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        # convert back into uint8, and make original image
        center = np.uint8(centers)
        res = center[labels.flatten()]
        res2 = res.reshape((image.shape))
        res2 = opencv2tensor(res2)
        return (res2,)

```
