---
tags:
- Image
---

# MonoMerge
## Documentation
- Class name: `MonoMerge`
- Category: `Bmad/image`
- Output node: `False`

The MonoMerge node is designed to merge two images into a monochromatic image based on a target color scheme (either towards white or black). This process involves comparing the luminance (L component) of corresponding pixels from both images and selecting the higher or lower value depending on the target, resulting in an image that emphasizes either the lighter or darker aspects of the combined images.
## Input types
### Required
- **`image1`**
    - The first image to be merged. It plays a crucial role in the merging process as its pixels are compared with those of the second image to determine the final image based on the target color scheme.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image2`**
    - The second image to be merged. It is compared against the first image to select the appropriate luminance values for the final monochromatic image, depending on the target color scheme.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`target`**
    - Specifies the target color scheme for the merge, either 'white' or 'black'. This determines whether the merge will favor lighter or darker pixel values from the input images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`output_format`**
    - Defines the format of the output image, such as JPEG, PNG, etc. This affects how the final image is encoded and saved.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting monochromatic image after merging the two input images based on the specified target color scheme.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MonoMerge:
    target = ["white", "black"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "target": (cls.target, {"default": "white"}),
                "output_format": (image_output_formats_options, {
                    "default": image_output_formats_options[0]
                })
                ,
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "monochromatic_merge"
    CATEGORY = images_category_path

    def monochromatic_merge(self, image1, image2, target, output_format):
        image1 = tensor2opencv(image1, 1)
        image2 = tensor2opencv(image2, 1)

        # Select the lesser L component at each pixel
        if target == "white":
            image = np.maximum(image1, image2)
        else:
            image = np.minimum(image1, image2)

        image = maybe_convert_img(image, 1, image_output_formats_options_map[output_format])
        image = opencv2tensor(image)

        return (image,)

```
