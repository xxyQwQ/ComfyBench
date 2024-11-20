---
tags:
- ImageResize
- ImageScaling
- ImageSize
- ImageTransformation
---

# Load & Resize Image
## Documentation
- Class name: `LoadAndResizeImage`
- Category: `KJNodes/image`
- Output node: `False`

This node is designed to load images from a specified path and resize them according to given dimensions, maintaining the aspect ratio if required. It supports handling of image masks, allowing for simultaneous resizing and optional adjustment of the mask channel. The node is capable of processing single images or batches, adapting to various image formats and ensuring compatibility with further image processing tasks.
## Input types
### Required
- **`image`**
    - The image or path to the image to be loaded and resized. It serves as the primary input for the node's operation, determining the content that will undergo resizing and potential mask adjustment.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Union[str, Image.Image]`
- **`resize`**
    - A boolean flag indicating whether the image should be resized. This parameter influences the node's decision to alter the image dimensions or maintain the original size.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`width`**
    - The desired width of the output image. It plays a crucial role in determining the final dimensions of the resized image, especially when maintaining aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The desired height of the output image. Similar to width, it affects the final dimensions of the resized image and is essential for aspect ratio calculations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`repeat`**
    - Indicates whether the image should be repeated to fill a larger canvas if necessary. This parameter is relevant for batch processing or specific layout requirements.
    - Comfy dtype: `INT`
    - Python dtype: `bool`
- **`keep_proportion`**
    - A flag to maintain the original aspect ratio of the image during resizing. It ensures that the resized image retains its proportional dimensions, avoiding distortion.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`divisible_by`**
    - Ensures the dimensions of the resized image are divisible by a specified number. This is often required for compatibility with certain image processing or machine learning models.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`mask_channel`**
    - Specifies the channel of the image to be used as a mask, allowing for selective processing or adjustment in conjunction with the main image resizing.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resized image, potentially alongside its mask, ready for further processing or analysis. This output is central to the node's purpose of preparing images for subsequent steps in a workflow.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The mask associated with the resized image, if applicable, providing additional data for selective processing or analysis.
    - Python dtype: `Optional[torch.Tensor]`
- **`width`**
    - Comfy dtype: `INT`
    - The width of the resized image, reflecting the final dimensions after processing.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the resized image, indicating the final dimensions post-resizing.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LoadAndResizeImage:
    _color_channels = ["alpha", "red", "green", "blue"]
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {
                    "image": (sorted(files), {"image_upload": True}),
                    "resize": ("BOOLEAN", { "default": False }),
                    "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                    "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                    "repeat": ("INT", { "default": 1, "min": 1, "max": 4096, "step": 1, }),
                    "keep_proportion": ("BOOLEAN", { "default": False }),
                    "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
                    "mask_channel": (s._color_channels, ), 
                    },
                }

    CATEGORY = "KJNodes/image"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT",)
    RETURN_NAMES = ("image", "mask", "width", "height",)
    FUNCTION = "load_image"

    def load_image(self, image, resize, width, height, repeat, keep_proportion, divisible_by, mask_channel):
        image_path = folder_paths.get_annotated_filepath(image)

        import node_helpers
        img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        W, H = img.size
        if resize:
            if keep_proportion:
                ratio = min(width / W, height / H)
                width = round(W * ratio)
                height = round(H * ratio)
            else:
                if width == 0:
                    width = W
                if height == 0:
                    height = H

            if divisible_by > 1:
                width = width - (width % divisible_by)
                height = height - (height % divisible_by)
        else:
            width, height = W, H

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            if resize:
                image = image.resize((width, height), Image.Resampling.BILINEAR)

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            mask = None
            c = mask_channel[0].upper()
            if c in i.getbands():
                if resize:
                    i = i.resize((width, height), Image.Resampling.BILINEAR)
                mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask)
                if c == 'A':
                     mask = 1. - mask
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")

            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
            if repeat > 1:
                output_image = output_image.repeat(repeat, 1, 1, 1)
                output_mask = output_mask.repeat(repeat, 1, 1)

     
        return (output_image, output_mask, width, height)
        

    @classmethod
    def IS_CHANGED(s, image):
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
