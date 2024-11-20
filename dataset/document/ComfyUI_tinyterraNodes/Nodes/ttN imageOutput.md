---
tags:
- Image
- ImageSave
---

# imageOutput
## Documentation
- Class name: `ttN imageOutput`
- Category: `🌏 tinyterra/image`
- Output node: `True`

The `ttN_imageOutput` node is designed for image processing and output management within the ComfyUI tinyterraNodes framework. It encapsulates functionality for handling image outputs, including saving images with specific configurations, adjusting image visibility based on user-defined parameters, and potentially transforming images for further use. This node plays a crucial role in the workflow by providing flexible output options and integrating with other components for image manipulation and storage.
## Input types
### Required
- **`image`**
    - The input image to be processed or managed by the node, serving as the primary data for output operations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image_output`**
    - Specifies the desired output behavior for images, such as whether to display, hide, or save the images, influencing how the node processes and returns image data.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`output_path`**
    - The file path where the image should be saved if the saving option is selected, providing a location for output storage.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`save_prefix`**
    - A prefix added to the saved image filename, allowing for organized naming conventions.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`number_padding`**
    - Determines the padding for numbering saved images, facilitating ordered file naming.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `int`
- **`file_type`**
    - The file type for the saved image, such as PNG or JPEG, defining the format of the output file.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`overwrite_existing`**
    - A boolean indicating whether existing files with the same name should be overwritten, controlling file management behavior.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`embed_workflow`**
    - Indicates whether the workflow metadata should be embedded within the image file, enhancing traceability.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image, returned based on the specified image output behavior.
    - Python dtype: `torch.Tensor`
- **`ui`**
    - Provides a UI component for displaying the processed images, if applicable, based on the specified image output behavior.
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ttN_imageOUPUT:
    version = '1.2.0'
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                "image": ("IMAGE",),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save"],{"default": "Preview"}),
                "output_path": ("STRING", {"default": folder_paths.get_output_directory(), "multiline": False}),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
                "number_padding": (["None", 2, 3, 4, 5, 6, 7, 8, 9],{"default": 5}),
                "file_type": (OUTPUT_FILETYPES, {"default": "png"}),
                "overwrite_existing": ("BOOLEAN", {"default": False}),
                "embed_workflow": ("BOOLEAN", {"default": True}),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                            "ttNnodeVersion": ttN_imageOUPUT.version},
            }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "output"
    CATEGORY = "🌏 tinyterra/image"
    OUTPUT_NODE = True

    def output(self, image, image_output, output_path, save_prefix, number_padding, file_type, overwrite_existing, embed_workflow, prompt, extra_pnginfo, my_unique_id):
        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo, number_padding, overwrite_existing, output_path)
        results = ttN_save.images(image, save_prefix, image_output, embed_workflow, file_type)

        if image_output in ("Hide", "Hide/Save"):
            return (image,)

        # Output image results to ui and node outputs
        return {"ui": {"images": results},
                "result": (image,)}

```
