---
tags:
- Image
- ImageLoad
---

# Load Image (as Mask) From URL
## Documentation
- Class name: `LoadImageAsMaskFromUrl`
- Category: `Art Venture/Image`
- Output node: `False`

This node is designed to load images from URLs and convert them into masks based on a specified channel. It supports selecting from alpha, red, green, or blue channels to create the mask, making it versatile for various image processing tasks where masking based on color channels is required.
## Input types
### Required
- **`image`**
    - A string containing the URL(s) of the image(s) to be loaded. If multiple URLs are provided, they should be separated by newlines. This parameter is essential for fetching the images from which masks will be created.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`channel`**
    - Specifies the color channel ('alpha', 'red', 'green', 'blue') to be used for creating the mask. This choice determines which part of the image data will be converted into a mask, affecting the outcome significantly.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`masks`**
    - Comfy dtype: `MASK`
    - The output is a tensor of masks created from the specified channel of the input images. These masks are suitable for further image processing tasks, such as compositing or inpainting.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class UtilLoadImageAsMaskFromUrl(UtilLoadImageFromUrl):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "channel": (["alpha", "red", "green", "blue"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)

    def load_image(self, image: str, channel: str, url=""):
        if not image or image == "":
            image = url

        urls = image.strip().split("\n")
        images, alphas = load_images_from_url([urls], False)

        masks = []

        for image, alpha in zip(images, alphas):
            if channel == "alpha":
                mask = alpha
            elif channel == "red":
                mask = image.getchannel("R")
            elif channel == "green":
                mask = image.getchannel("G")
            elif channel == "blue":
                mask = image.getchannel("B")

            mask = np.array(mask).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)

            masks.append(mask)

        return (torch.stack(masks, dim=0),)

```
