---
tags:
- Latent
- LatentPreview
---

# 🔧 Image Preview From Latent
## Documentation
- Class name: `ImagePreviewFromLatent+`
- Category: `essentials/image utils`
- Output node: `True`

This node is designed to generate a preview image from a given latent representation. It leverages specific decoding techniques to transform latent vectors into visual previews, facilitating a visual understanding of the latent space.
## Input types
### Required
- **`latent`**
    - The latent representation to be transformed into a preview image. This input is crucial for visualizing the encoded information in a human-interpretable format.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - The variational autoencoder model used for decoding the latent representation into an image.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
- **`tile_size`**
    - Specifies the size of the tiles used in the decoding process, affecting the resolution of the output image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`image`**
    - An optional input image that can be used in conjunction with the latent representation for generating the preview.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `PIL.Image.Image`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The generated preview image, offering a visual interpretation of the latent representation.
    - Python dtype: `PIL.Image.Image`
- **`MASK`**
    - Comfy dtype: `MASK`
    - An optional mask that may be generated alongside the preview image, depending on the node's implementation.
    - Python dtype: `PIL.Image.Image`
- **`width`**
    - Comfy dtype: `INT`
    - The width of the generated preview image.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the generated preview image.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImagePreviewFromLatent(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE", ),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64})
            }, "optional": {
                "image": (["none"], {"image_upload": False}),
            }, "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, latent, vae, tile_size, prompt=None, extra_pnginfo=None, image=None, filename_prefix="ComfyUI"):
        mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        ui = None

        if image.startswith("clipspace"):
            image_path = folder_paths.get_annotated_filepath(image)
            if not os.path.exists(image_path):
                raise ValueError(f"Clipspace image does not exist anymore, select 'none' in the image field.")

            img = pillow(Image.open, image_path)
            img = pillow(ImageOps.exif_transpose, img)
            if img.mode == "I":
                img = img.point(lambda i: i * (1 / 255))
            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in img.getbands():
                mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            ui = {
                "filename": os.path.basename(image_path),
                "subfolder": os.path.dirname(image_path),
                "type": "temp",
            }
        else:
            if tile_size > 0:
                tile_size = max(tile_size, 320)
                image = vae.decode_tiled(latent["samples"], tile_x=tile_size // 8, tile_y=tile_size // 8, )
            else:
                image = vae.decode(latent["samples"])
            ui = self.save_images(image, filename_prefix, prompt, extra_pnginfo)

        out = {**ui, "result": (image, mask, image.shape[2], image.shape[1],)}
        return out

```
