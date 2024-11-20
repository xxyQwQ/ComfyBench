---
tags:
- Mask
---

# Painter Node
## Documentation
- Class name: `PainterNode`
- Category: `AlekPet Nodes/image`
- Output node: `False`

The PainterNode is designed to facilitate image processing and manipulation within a custom node environment. It enables the dynamic updating of images based on user input, supports asynchronous communication with a server for image updates, and manages node-specific settings for personalized image processing workflows.
## Input types
### Required
- **`image`**
    - The primary image input for processing. It serves as the base for any transformations or updates applied by the node.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `torch.Tensor`
### Optional
- **`images`**
    - An optional list of images to be processed or updated. These images can be used for batch processing or as additional inputs for the node's operations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`update_node`**
    - A flag indicating whether the node should update its internal state based on the provided images. This affects how images are processed and integrated into the node's workflow.
    - Comfy dtype: `COMBO[BOOLEAN]`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image after applying the node's operations, ready for further use or display.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - An optional mask generated during the image processing, which can be used for further image manipulation or analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [Image Batch](../../was-node-suite-comfyui/Nodes/Image Batch.md)
    - [ImageScale](../../Comfy/Nodes/ImageScale.md)



## Source code
```python
class PainterNode(object):

    @classmethod
    def INPUT_TYPES(self):
        self.canvas_set = False

        work_dir = folder_paths.get_input_directory()
        imgs = [
            img
            for img in os.listdir(work_dir)
            if os.path.isfile(os.path.join(work_dir, img))
        ]

        return {
            "required": {"image": (sorted(imgs),)},
            "hidden": {"unique_id": "UNIQUE_ID"},
            "optional": {"images": ("IMAGE",), "update_node": (([True, False],))},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "painter_execute"

    CATEGORY = "AlekPet Nodes/image"

    def painter_execute(self, image, unique_id, update_node=True, images=None):
        # Piping image input
        if unique_id not in PAINTER_DICT:
            PAINTER_DICT[unique_id] = self

        if update_node == True and images is not None:

            input_images = []

            for imgs in images:
                i = 255.0 * imgs.cpu().numpy()
                i = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                input_images.append(toBase64ImgUrl(i))

            PAINTER_DICT[unique_id].canvas_set = False

            PromptServer.instance.send_sync(
                "alekpet_get_image", {"unique_id": unique_id, "images": input_images}
            )
            if not asyncio.run(wait_canvas_change(unique_id)):
                print(f"Painter_{unique_id}: Failed to get image!")
            else:
                print(f"Painter_{unique_id}: Image received, canvas changed!")
        # end - Piping image input

        image_path = folder_paths.get_annotated_filepath(image)

        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if "A" in i.getbands():
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return (image, mask.unsqueeze(0))

    @classmethod
    def IS_CHANGED(self, image, unique_id, update_node=True, images=None):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())

        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(self, image, unique_id, update_node=True, images=None):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

```
