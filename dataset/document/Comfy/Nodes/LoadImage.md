---
tags:
- Image
- ImageLoad
---

# Load Image
## Documentation
- Class name: `LoadImage`
- Category: `image`
- Output node: `False`

The LoadImage node is designed for loading and processing images from specified paths. It handles various image formats, applies necessary transformations like EXIF orientation correction, and converts images to a consistent format and tensor representation for further processing in machine learning pipelines.
## Input types
### Required
- **`image`**
    - The 'image' parameter specifies the image file to be loaded and processed from a predefined list of available files. It is crucial for determining the input for the node's operations, affecting the subsequent image manipulation and conversion steps.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - A tensor representing the loaded and processed image, ready for use in further processing or model inference.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - A tensor representing the mask associated with the input image, if applicable, indicating areas of interest or regions to ignore.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [SVD_img2vid_Conditioning](../../Comfy/Nodes/SVD_img2vid_Conditioning.md)
    - [ReActorFaceSwap](../../comfyui-reactor-node/Nodes/ReActorFaceSwap.md)
    - IPAdapterApply
    - Reroute
    - [VAEEncodeForInpaint](../../Comfy/Nodes/VAEEncodeForInpaint.md)
    - [PrepImageForClipVision](../../ComfyUI_IPAdapter_plus/Nodes/PrepImageForClipVision.md)
    - [MiDaS-DepthMapPreprocessor](../../comfyui_controlnet_aux/Nodes/MiDaS-DepthMapPreprocessor.md)
    - [OpenposePreprocessor](../../comfyui_controlnet_aux/Nodes/OpenposePreprocessor.md)
    - [InpaintPreprocessor](../../comfyui_controlnet_aux/Nodes/InpaintPreprocessor.md)
    - [VAEEncode](../../Comfy/Nodes/VAEEncode.md)



## Source code
```python
class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        
        img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
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
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
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

        return (output_image, output_mask)

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
