---
tags:
- Image
---

# Load Image From URL
## Documentation
- Class name: `LoadImageFromUrl`
- Category: `Art Venture/Image`
- Output node: `False`

This node is designed to load images from URLs, supporting a variety of formats including direct links, file paths, and data URIs. It abstracts the complexities of fetching and decoding images from different sources, making it easier to integrate external images into workflows or applications.
## Input types
### Required
### Optional
- **`image`**
    - The image data directly provided for processing, allowing for flexibility in sourcing images beyond URLs.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`keep_alpha_channel`**
    - Determines whether the alpha channel of the image should be preserved during processing, affecting the output image format.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`output_mode`**
    - Specifies the desired format of the output image, influencing how the image is processed and returned.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `str`
- **`url`**
    - The URL or path of the image to be loaded. It supports direct links, file paths, and data URIs, enabling the node to fetch and decode images from various sources.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - The loaded images, processed according to the specified parameters such as alpha channel preservation and output mode.
    - Python dtype: `List[PIL.Image.Image]`
- **`masks`**
    - Comfy dtype: `MASK`
    - The masks generated for the images, if any, based on the processing options selected.
    - Python dtype: `List[torch.Tensor]`
- **`has_image`**
    - Comfy dtype: `BOOLEAN`
    - A flag indicating whether an image was successfully loaded and processed.
    - Python dtype: `bool`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class UtilLoadImageFromUrl:
    def __init__(self) -> None:
        self.output_dir = folder_paths.get_temp_directory()
        self.filename_prefix = "TempImageFromUrl"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "image": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "keep_alpha_channel": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
                "output_mode": (
                    "BOOLEAN",
                    {"default": False, "label_on": "list", "label_off": "batch"},
                ),
                "url": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    OUTPUT_IS_LIST = (True, True, False)
    RETURN_NAMES = ("images", "masks", "has_image")
    CATEGORY = "Art Venture/Image"
    FUNCTION = "load_image"

    def load_image(self, image="", keep_alpha_channel=False, output_mode=False, url=""):
        if not image or image == "":
            image = url

        urls = image.strip().split("\n")
        images, masks = load_images_from_url(urls, keep_alpha_channel)
        if len(images) == 0:
            image = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            return ([image], [mask], False)

        previews = []
        np_images = []
        np_masks = []

        for image, mask in zip(images, masks):
            # save image to temp folder
            preview = prepare_image_for_preview(image, self.output_dir, self.filename_prefix)
            image = pil2tensor(image)

            if mask:
                mask = np.array(mask).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            previews.append(preview)
            np_images.append(image)
            np_masks.append(mask.unsqueeze(0))

        if output_mode:
            result = (np_images, np_masks, True)
        else:
            has_size_mismatch = False
            if len(np_images) > 1:
                for image in np_images[1:]:
                    if image.shape[1] != np_images[0].shape[1] or image.shape[2] != np_images[0].shape[2]:
                        has_size_mismatch = True
                        break

            if has_size_mismatch:
                raise Exception("To output as batch, images must have the same size. Use list output mode instead.")

            result = ([torch.cat(np_images)], [torch.cat(np_masks)], True)

        return {"ui": {"images": previews}, "result": result}

```
