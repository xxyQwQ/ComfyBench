# ∞ Export Images and Captions
## Documentation
- Class name: `SaltExportCaptions`
- Category: `SALT/IO`
- Output node: `True`

The `SaltExportCaptions` node is designed for exporting pairs of images and their corresponding captions to a specified folder. It ensures that each image is saved alongside its caption in a text file, facilitating organized storage and retrieval of visual-textual data.
## Input types
### Required
- **`images`**
    - A list of images to be exported. Each image is processed and saved in the specified folder along with its caption.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`captions`**
    - A list of text captions corresponding to the images. Each caption is saved in a text file named after its image.
    - Comfy dtype: `LIST`
    - Python dtype: `List[str]`
- **`folder`**
    - The name of the folder where the images and captions will be exported. If the folder does not exist, it will be created.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltExportCaptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "captions": ("LIST",),
                "folder": ("STRING", {"default": "folder_name"}),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True

    FUNCTION = "export"
    CATEGORY = f"{MENU_NAME}/IO"

    def export(self, images, captions, folder):
        output = folder_paths.get_output_directory()
        base_path = os.path.abspath(output)
        target_path = os.path.abspath(os.path.join(base_path, folder))

        if not target_path.startswith(base_path):
            raise ValueError(f"Invalid folder path `{target_path}`")

        os.makedirs(target_path, exist_ok=True)

        if len(images) != len(captions):
            raise ValueError("The number of images and captions must be equal.")

        for i, (tensor, string) in enumerate(zip(images, captions)):
            image = tensor2pil(tensor)
            image_path = os.path.join(target_path, f"image_{i}.png")
            text_path = os.path.join(target_path, f"image_{i}.txt")

            image.save(image_path)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(string)

        return ()

```
