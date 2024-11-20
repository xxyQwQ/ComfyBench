---
tags:
- Face
---

# Face Swap (mtb)
## Documentation
- Class name: `Face Swap (mtb)`
- Category: `mtb/facetools`
- Output node: `False`

Performs face swapping between a source and a target image using deep learning models, specifically designed to handle complex scenarios involving multiple faces and preserving facial features accurately.
## Input types
### Required
- **`image`**
    - The image tensor where the face(s) will be swapped onto. It serves as the canvas for the operation, determining the final output's visual context.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`reference`**
    - The reference image tensor providing the face(s) to be swapped into the target image. It acts as the source of facial features for the swap.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`faces_index`**
    - A string specifying the indices of faces in the target image to be swapped. It allows selective swapping, enhancing control over the output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`faceanalysis_model`**
    - The model used for analyzing faces within images, crucial for identifying and extracting facial features accurately.
    - Comfy dtype: `FACE_ANALYSIS_MODEL`
    - Python dtype: `object`
- **`faceswap_model`**
    - The model responsible for the actual face swapping process, leveraging deep learning to ensure realistic and seamless swaps.
    - Comfy dtype: `FACESWAP_MODEL`
    - Python dtype: `object`
### Optional
- **`preserve_alpha`**
    - A boolean indicating whether to preserve the alpha channel of the image, allowing for transparency handling in images with RGBA format.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after the face swap has been performed, showcasing the swapped faces within the original image's context.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MTB_FaceSwap:
    """Face swap using deepinsight/insightface models"""

    model = None
    model_path = None

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "faces_index": ("STRING", {"default": "0"}),
                "faceanalysis_model": (
                    "FACE_ANALYSIS_MODEL",
                    {"default": "None"},
                ),
                "faceswap_model": ("FACESWAP_MODEL", {"default": "None"}),
            },
            "optional": {
                "preserve_alpha": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "swap"
    CATEGORY = "mtb/facetools"

    def swap(
        self,
        image: torch.Tensor,
        reference: torch.Tensor,
        faces_index: str,
        faceanalysis_model,
        faceswap_model,
        preserve_alpha=False,
    ):
        def do_swap(img):
            model_management.throw_exception_if_processing_interrupted()
            img = tensor2pil(img)[0]
            ref = tensor2pil(reference)[0]

            alpha_channel = None
            if preserve_alpha and img.mode == "RGBA":
                alpha_channel = img.getchannel("A")
                img = img.convert("RGB")

            face_ids = {
                int(x)
                for x in faces_index.strip(",").split(",")
                if x.isnumeric()
            }
            sys.stdout = NullWriter()
            swapped = swap_face(
                faceanalysis_model, ref, img, faceswap_model, face_ids
            )
            sys.stdout = sys.__stdout__
            if alpha_channel:
                swapped.putalpha(alpha_channel)
            return pil2tensor(swapped)

        batch_count = image.size(0)

        log.info(f"Running insightface swap (batch size: {batch_count})")

        if reference.size(0) != 1:
            raise ValueError("Reference image must have batch size 1")
        if batch_count == 1:
            image = do_swap(image)

        else:
            image_batch = [do_swap(image[i]) for i in range(batch_count)]
            image = torch.cat(image_batch, dim=0)

        return (image,)

```
