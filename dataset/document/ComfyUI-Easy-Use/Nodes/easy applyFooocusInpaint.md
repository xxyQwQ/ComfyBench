---
tags:
- DepthMap
- Image
- Inpaint
---

# Easy Apply Fooocus Inpaint
## Documentation
- Class name: `easy applyFooocusInpaint`
- Category: `EasyUse/Inpaint`
- Output node: `False`

The `applyFooocusInpaint` node is designed to integrate inpainting capabilities into a model's pipeline by applying specific inpainting heads and patches. This process involves loading and applying pre-trained components to modify the model in a way that it can perform inpainting tasks, which is the process of filling in missing or corrupted parts of images.
## Input types
### Required
- **`model`**
    - The model parameter represents the base model to which inpainting capabilities will be added. It is crucial for defining the starting point of the inpainting process.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`latent`**
    - The latent parameter contains the encoded representation of the image to be inpainted, serving as a critical input for the inpainting process.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`head`**
    - The head parameter specifies the inpainting head to be used, determining the specific method and characteristics of the inpainting process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`patch`**
    - The patch parameter indicates the specific inpainting patch to apply, further customizing the inpainting behavior.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model with inpainting capabilities integrated, ready for performing inpainting tasks.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class applyFooocusInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "head": (list(FOOOCUS_INPAINT_HEAD.keys()),),
                "patch": (list(FOOOCUS_INPAINT_PATCH.keys()),),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY = "EasyUse/Inpaint"
    FUNCTION = "apply"

    def apply(self, model, latent, head, patch):

        head_file = get_local_filepath(FOOOCUS_INPAINT_HEAD[head]["model_url"], INPAINT_DIR)
        inpaint_head_model = InpaintHead()
        sd = torch.load(head_file, map_location='cpu')
        inpaint_head_model.load_state_dict(sd)

        patch_file = get_local_filepath(FOOOCUS_INPAINT_PATCH[patch]["model_url"], INPAINT_DIR)
        inpaint_lora = comfy.utils.load_torch_file(patch_file, safe_load=True)

        patch = (inpaint_head_model, inpaint_lora)
        worker = InpaintWorker(node_name="easy kSamplerInpainting")
        cloned = model.clone()

        m, = worker.patch(cloned, latent, patch)
        return (m,)

```
