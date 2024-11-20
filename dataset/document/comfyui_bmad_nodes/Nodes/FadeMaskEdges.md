---
tags:
- Mask
---

# FadeMaskEdges
## Documentation
- Class name: `FadeMaskEdges`
- Category: `Bmad/CV/Misc`
- Output node: `False`

This node is designed for refining the edges of masks in images, specifically targeting the enhancement of edge blending and smooth transitions. It focuses on premultiplying and alpha blending the edges of a subject to prevent the intrusion of outer pixels, which is particularly useful for stylized subjects or images requiring precise edge manipulation.
## Input types
### Required
- **`binary_image`**
    - Represents the mask image to be processed, where the edge modifications will be applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`edge_size`**
    - Defines the size of the edge to be faded, impacting the extent of the edge modification.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`edge_tightness`**
    - Controls the rate at which the edge fades to black, affecting the sharpness of the edge transition.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`edge_exponent`**
    - Determines the nature of the fade, potentially softening small lines for a smoother transition.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`smoothing_diameter`**
    - Specifies the diameter for optional smoothing applied after edge processing, to further refine the edge appearance.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`paste_original_blacks`**
    - A boolean flag indicating whether to paste original black values back into the image, preserving dark regions.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image with enhanced and smoothly blended edges.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FadeMaskEdges:
    """
    The original intent is to premultiply and alpha blend a subject's edges to avoid outer pixels creeping in.

    A very slight blur near the edges afterwards when using paste_original_blacks and low tightness may be required,
     but this should be done after premultiplying and setting the alpha.

    Stylized subject's, such as drawings with black outlines, may benefit from using different 2 edge fades:
        1. a fade with higher edge size for the premultiplication, fading the subject into blackness
        2. a tighter fade for the alpha
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_image": ("IMAGE",),
                "edge_size": ("FLOAT", {"default": 5.0, "min": 1.0, "step": 1.0}),
                # how quick does it fade to black
                "edge_tightness": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 10.0, "step": 0.05}),
                # how does it fade, may be used to weaken small lines; 1 = linear transition
                "edge_exponent": ("FLOAT", {"default": 1, "min": 0.1, "max": 10.0, "step": 0.1}),
                "smoothing_diameter": ("INT", {"default": 10, "min": 2, "max": 256, "step": 1}),
                "paste_original_blacks": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = f"{cv_category_path}/Misc"

    def apply(self, binary_image, edge_size, edge_tightness, edge_exponent, smoothing_diameter, paste_original_blacks):
        binary_image = tensor2opencv(binary_image, 1)
        # _, binary_image = cv.threshold(gray_image, 128, 255, cv.THRESH_BINARY) # suppose it's already binary

        # compute L2 (euclidean) distance -> normalize with respect to edge size -> smooth
        distance_transform = cv.distanceTransform(binary_image, cv.DIST_L2, cv.DIST_MASK_3)
        normalized_distance = distance_transform / edge_size
        smoothed_distance = cv.bilateralFilter(normalized_distance, smoothing_diameter, 75, 75)

        # darken the white pixels based on smoothed distance and "edge tightness"
        diff = 1 - smoothed_distance
        darkened_image = (abs(diff * edge_tightness) ** (1 / edge_exponent)) * np.sign(diff)
        darkened_image = np.clip(darkened_image, 0, 1)
        darkened_image = (darkened_image * 255).astype(np.uint8)

        if paste_original_blacks:  # mask original black pixels
            black_mask = binary_image < 1
            darkened_image[black_mask] = 0

        output_image = binary_image - darkened_image  # darken original image
        output_image = opencv2tensor(output_image)
        return (output_image,)

```
