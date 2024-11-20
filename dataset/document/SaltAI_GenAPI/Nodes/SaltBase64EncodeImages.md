# Base64 Encode Images
## Documentation
- Class name: `SaltBase64EncodeImages`
- Category: `SALT/Image/Process`
- Output node: `False`

The `SaltBase64EncodeImages` node is designed to convert a collection of images, either passed directly as tensors or specified via file paths or URLs, into their Base64-encoded string representations. This process involves resizing the images to specified dimensions and encoding them, facilitating their use in environments where binary image data needs to be embedded or transmitted in text form.
## Input types
### Required
- **`width`**
    - Specifies the desired width to which the images should be resized. This parameter plays a crucial role in determining the final dimensions of the encoded images, affecting their appearance and size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Determines the desired height for the image resizing process. Similar to width, it significantly influences the encoded images' dimensions and visual quality.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`images`**
    - An optional tensor containing images to be encoded. When provided, it bypasses the need for loading images from paths or URLs, directly converting the tensor images to Base64 strings.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`paths_or_urls`**
    - A string containing either file paths or URLs, separated by newlines, from which images will be loaded, resized, and encoded. This allows for flexible sourcing of images from local storage or the web.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`base64_images`**
    - Comfy dtype: `LIST`
    - A list of strings, each representing a Base64-encoded image. This output facilitates the embedding or transmission of images in environments where binary data is not supported.
    - Python dtype: `List[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltBase64EncodeImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 256, "min": 1}),
                "height": ("INT", {"default": 256, "min": 1}),
            },
            "optional": {
                "images": ("IMAGE",),
                "paths_or_urls": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("base64_images",)

    FUNCTION = "load_images"
    CATEGORY = f"{MENU_NAME}/Image/Process"

    def load_images(self, width, height, images=None, paths_or_urls=""):
        source_list = paths_or_urls.splitlines()
        base64_images = []

        if isinstance(images, torch.Tensor):
            for tensor in images:
                image = tensor2pil(tensor)
                img_str = base64_encode_image(image, width, height)
                base64_images.append(img_str)

        if paths_or_urls.strip():
            for source in source_list:
                source = source.strip()
                if not source:
                    continue

                try:
                    if source.startswith("http://") or source.startswith("https://"):
                        response = requests.get(source)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content))
                    elif os.path.exists(source):
                        image = Image.open(source)
                    else:
                        continue
                except (requests.exceptions.RequestException, IOError) as e:
                    print(f"Error loading image from {source}: {e}")
                    continue

                img_str = base64_encode_image(image, width, height)
                base64_images.append(img_str)

        if not base64_images:
            raise ValueError("Unable to resolve any images from `sources` input.")

        return (base64_images,)

```
