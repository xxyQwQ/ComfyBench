---
tags:
- Image
- ImageLoad
---

# Load Image (as Mask)
## Documentation
- Class name: `LoadImageMask`
- Category: `mask`
- Output node: `False`

The LoadImageMask node is designed for loading and processing an image to generate a mask based on a specified color channel. It supports handling images with different formats and orientations, converting them to a standardized RGBA format if necessary, and extracting a specific channel (alpha, red, green, blue) to create a mask. This functionality is crucial for tasks that require image segmentation or specific channel manipulation.
## Input types
### Required
- **`image`**
    - Specifies the image file to be loaded and processed. The selection is limited to the files available in a predefined input directory, facilitating the upload and handling of images for mask generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`channel`**
    - Determines the color channel ('alpha', 'red', 'green', 'blue') of the image to be used for mask creation. This choice directly influences the resulting mask by selecting the relevant channel data.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a mask generated from the specified color channel of the input image. This mask is useful for segmentation and other image processing tasks that require specific channel information.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LoadImageMask:
    _color_channels = ["alpha", "red", "green", "blue"]
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True}),
                     "channel": (s._color_channels, ), }
                }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image"
    def load_image(self, image, channel):
        image_path = folder_paths.get_annotated_filepath(image)
        i = node_helpers.pillow(Image.open, image_path)
        i = node_helpers.pillow(ImageOps.exif_transpose, i)
        if i.getbands() != ("R", "G", "B", "A"):
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            i = i.convert("RGBA")
        mask = None
        c = channel[0].upper()
        if c in i.getbands():
            mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
            if c == 'A':
                mask = 1. - mask
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (mask.unsqueeze(0),)

    @classmethod
    def IS_CHANGED(s, image, channel):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

```
