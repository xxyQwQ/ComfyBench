---
tags:
- Image
- Multimedia
---

# Image Dir Iterator
## Documentation
- Class name: `ImageDirIterator`
- Category: `cspnodes`
- Output node: `False`

The ImageDirIterator node is designed to iterate through images in a specified directory, providing functionality to access and manipulate image files based on directory structure and file properties. This node is likely to facilitate operations such as image retrieval, sorting, and processing, aiming to streamline workflows that involve handling multiple images stored in a directory.
## Input types
### Required
- **`directory_path`**
    - Specifies the path to the directory containing images. This path is used to locate and iterate through the image files for processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`image_index`**
    - An index to specify which image to retrieve from the sorted list of images in the directory. This allows for sequential or specific access to images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The image retrieved from the directory at the specified index, processed and returned as a tensor.
    - Python dtype: `torch.Tensor`
- **`string`**
    - Comfy dtype: `STRING`
    - The filename of the retrieved image, without its extension, providing a means to identify or reference the image.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImageDirIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {}),
                "image_index": ("INT", {"default": 0})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")

    FUNCTION = "get_image_by_index"

    CATEGORY = "cspnodes"

    def get_image_by_index(self, directory_path, image_index):
        # Get list of image files sorted by modification time (most recent first)
        image_files = sorted(
            [os.path.join(directory_path, f) for f in os.listdir(directory_path)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))],
            key=lambda x: os.path.getmtime(x), reverse=True
        )

        # Wrap the index around using modulo
        image_index = image_index % len(image_files)

        # Load and preprocess the image
        image = Image.open(image_files[image_index])
        image = ImageOps.exif_transpose(image)  # Correct orientation
        image = image.convert("RGB")  # Ensure image is in RGB format

        # Convert image to tensor
        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

        # Get the filename without extension and remove quotes
        filename_without_ext = os.path.splitext(os.path.basename(image_files[image_index]))[0]
        filename_without_ext = filename_without_ext.encode('utf-8').decode('unicode_escape')

        return (image_tensor, filename_without_ext)

```
