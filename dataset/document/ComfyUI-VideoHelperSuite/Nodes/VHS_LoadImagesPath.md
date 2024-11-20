---
tags:
- Image
- ImageLoad
---

# Load Images (Path) 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_LoadImagesPath`
- Category: `Video Helper Suite 🎥🅥🅗🅢`
- Output node: `False`

This node is designed for loading a batch of images from a specified directory path. It streamlines the process of importing multiple images into the workflow, facilitating batch processing and manipulation of image data within the video helper suite.
## Input types
### Required
- **`directory`**
    - The directory path from which images are to be loaded. This parameter is crucial for specifying the source of the images, thereby enabling the node to retrieve and process the image files located at the given path.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`image_load_cap`**
    - Limits the number of images to be loaded from the directory, allowing for control over the batch size for processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`skip_first_images`**
    - Skips a specified number of images from the beginning of the directory, useful for starting the loading process from a certain point.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`select_every_nth`**
    - Loads every nth image from the directory, providing a means to sample the images at a specified interval.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`meta_batch`**
    - A reference to a batch manager object that handles meta information about the batch processing, such as tracking the progress and managing resources.
    - Comfy dtype: `VHS_BatchManager`
    - Python dtype: `BatchManager`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - A batch of images loaded from the specified directory path. This output is essential for subsequent image processing or analysis tasks, providing a collection of images ready for manipulation or examination.
    - Python dtype: `torch.Tensor`
- **`MASK`**
    - Comfy dtype: `MASK`
    - A batch of masks corresponding to the loaded images, used for image segmentation or editing tasks.
    - Python dtype: `torch.Tensor`
- **`frame_count`**
    - Comfy dtype: `INT`
    - The total number of frames (images) loaded from the directory, indicating the size of the batch processed.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ImpactImageBatchToImageList](../../ComfyUI-Impact-Pack/Nodes/ImpactImageBatchToImageList.md)
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [IPAdapterEncoder](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterEncoder.md)
    - LinearBatchCreativeInterpolation



## Source code
```python
class LoadImagesFromDirectoryPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/images", "vhs_path_extensions": []}),
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
        directory = strip_path(directory)
        if directory is None or validate_load_images(directory) != True:
            raise Exception("directory is not valid: " + directory)

        return load_images(directory, **kwargs)
    
    @classmethod
    def IS_CHANGED(s, directory: str, **kwargs):
        if directory is None:
            return "input"
        return is_changed_load_images(directory, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(s, directory: str, **kwargs):
        if directory is None:
            return True
        return validate_load_images(strip_path(directory))

```
