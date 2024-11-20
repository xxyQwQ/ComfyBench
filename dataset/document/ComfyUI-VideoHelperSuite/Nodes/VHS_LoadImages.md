---
tags:
- Image
- ImageLoad
---

# Load Images (Upload) 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_LoadImages`
- Category: `Video Helper Suite 🎥🅥🅗🅢`
- Output node: `False`

The node 'VHS_LoadImages' is designed for loading a batch of images from a directory, processing them according to specified parameters such as image load cap, skipping initial images, and selecting every nth image. It facilitates the handling of image data for further processing or analysis within a video helper suite, streamlining the workflow for video and image manipulation tasks.
## Input types
### Required
- **`directory`**
    - Specifies the directory path from which images are to be loaded. It is crucial for determining the source of the images to be processed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`image_load_cap`**
    - Limits the number of images to be loaded from the directory, optimizing resource usage and processing time.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`skip_first_images`**
    - Skips a specified number of initial images in the directory, allowing for flexible data handling and selection.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`select_every_nth`**
    - Loads every nth image from the directory, providing a mechanism for sampling or reducing the dataset size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`meta_batch`**
    - Optional parameter for integrating with a meta batch processing system, enabling batch processing of images.
    - Comfy dtype: `VHS_BatchManager`
    - Python dtype: `VHS_BatchManager`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The processed images loaded from the directory, ready for further manipulation or analysis.
    - Python dtype: `torch.Tensor`
- **`MASK`**
    - Comfy dtype: `MASK`
    - Generated masks for the loaded images, applicable when images have an alpha channel for transparency.
    - Python dtype: `torch.Tensor`
- **`frame_count`**
    - Comfy dtype: `INT`
    - The total number of images successfully loaded and processed.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - Reroute
    - [ScribblePreprocessor](../../comfyui_controlnet_aux/Nodes/ScribblePreprocessor.md)
    - [EmptyLatentImage](../../Comfy/Nodes/EmptyLatentImage.md)
    - [DWPreprocessor](../../comfyui_controlnet_aux/Nodes/DWPreprocessor.md)
    - [TilePreprocessor](../../comfyui_controlnet_aux/Nodes/TilePreprocessor.md)
    - [LineArtPreprocessor](../../comfyui_controlnet_aux/Nodes/LineArtPreprocessor.md)
    - [Zoe-DepthMapPreprocessor](../../comfyui_controlnet_aux/Nodes/Zoe-DepthMapPreprocessor.md)



## Source code
```python
class LoadImagesFromDirectoryUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        directories = []
        for item in os.listdir(input_dir):
            if not os.path.isfile(os.path.join(input_dir, item)) and item != "clipspace":
                directories.append(item)
        return {
            "required": {
                "directory": (directories,),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                "meta_batch": ("VHS_BatchManager",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "frame_count")
    FUNCTION = "load_images"

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    def load_images(self, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(strip_path(directory))
        return load_images(directory, **kwargs)
    
    @classmethod
    def IS_CHANGED(s, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(strip_path(directory))
        return is_changed_load_images(directory, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(s, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(strip_path(directory))
        return validate_load_images(directory)

```
