---
tags:
- Image
- ImageLoad
---

# 🚫Load Images [DEPRECATED] 🛂🅐🅒🅝
## Documentation
- Class name: `LoadImagesFromDirectory`
- Category: ``
- Output node: `False`

This node is designed for loading and processing a batch of images from a specified directory, including their associated masks if available. It supports filtering and limiting the number of images loaded based on provided criteria, and prepares the images and masks for further processing or analysis.
## Input types
### Required
- **`directory`**
    - Specifies the directory path from which images are to be loaded. It is crucial for locating and accessing the image files for processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`image_load_cap`**
    - Determines the maximum number of images to load from the directory. If set to 0, there is no limit on the number of images to load.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`start_index`**
    - Specifies the index from which to start loading images within the directory, allowing for partial loading of images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - A tensor containing the loaded images, concatenated along the first dimension.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - A tensor containing the masks associated with the loaded images, stacked along the first dimension.
    - Python dtype: `torch.Tensor`
- **`int`**
    - Comfy dtype: `INT`
    - The total number of images successfully loaded and processed.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LoadImagesFromDirectory:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    FUNCTION = "load_images"

    CATEGORY = ""

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory} cannot be found.'")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]
        # start at start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            images.append(image)
            masks.append(mask)
            image_count += 1
        
        if len(images) == 0:
            raise FileNotFoundError(f"No images could be loaded from directory '{directory}'.")

        return (torch.cat(images, dim=0), torch.stack(masks, dim=0), image_count)

```
