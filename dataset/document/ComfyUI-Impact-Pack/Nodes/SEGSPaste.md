---
tags:
- Image
- ImageComposite
---

# SEGSPaste
## Documentation
- Class name: `SEGSPaste`
- Category: `ImpactPack/Detailer`
- Output node: `False`

This node specializes in blending and manipulating image segments, leveraging latent space representations to achieve seamless integration of visual elements. It focuses on the precise control over the placement and blending of image segments within a given scene, enhancing the overall composition through advanced conditioning techniques.
## Input types
### Required
- **`image`**
    - A string path or identifier for an image that is to be blended or manipulated within the scene. It serves as a secondary input for visual reference or direct integration.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`segs`**
    - Represents the segments to be manipulated or blended, providing the primary elements for visual composition within the node.
    - Comfy dtype: `SEGS`
    - Python dtype: `torch.Tensor`
- **`feather`**
    - Defines the feathering amount for the edges of the segments, allowing for smoother integration into the target scene.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`alpha`**
    - Specifies the opacity level of the segments, enabling fine-tuned control over their visibility and blending with the background.
    - Comfy dtype: `INT`
    - Python dtype: `float`
### Optional
- **`ref_image_opt`**
    - Optional. A reference image that can be used for additional context or alignment purposes during the manipulation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Optional[str]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - Outputs the modified image after the blending and manipulation processes, incorporating the segments seamlessly into the scene.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [VHS_VideoCombine](../../ComfyUI-VideoHelperSuite/Nodes/VHS_VideoCombine.md)
    - [ImageUpscaleWithModel](../../Comfy/Nodes/ImageUpscaleWithModel.md)



## Source code
```python
class SEGSPaste:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "segs": ("SEGS", ),
                     "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                     "alpha": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                     },
                "optional": {"ref_image_opt": ("IMAGE", ), }
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    @staticmethod
    def doit(image, segs, feather, alpha=255, ref_image_opt=None):

        segs = core.segs_scale_match(segs, image.shape)

        result = None
        for i, single_image in enumerate(image):
            image_i = single_image.unsqueeze(0).clone()

            for seg in segs[1]:
                ref_image = None
                if ref_image_opt is None and seg.cropped_image is not None:
                    cropped_image = seg.cropped_image
                    if isinstance(cropped_image, np.ndarray):
                        cropped_image = torch.from_numpy(cropped_image)
                    ref_image = cropped_image[i].unsqueeze(0)
                elif ref_image_opt is not None:
                    ref_tensor = ref_image_opt[i].unsqueeze(0)
                    ref_image = crop_image(ref_tensor, seg.crop_region)
                if ref_image is not None:
                    if seg.cropped_mask.ndim == 3 and len(seg.cropped_mask) == len(image):
                        mask = seg.cropped_mask[i]
                    elif seg.cropped_mask.ndim == 3 and len(seg.cropped_mask) > 1:
                        print(f"[Impact Pack] WARN: SEGSPaste - The number of the mask batch({len(seg.cropped_mask)}) and the image batch({len(image)}) are different. Combine the mask frames and apply.")
                        combined_mask = (seg.cropped_mask[0] * 255).to(torch.uint8)

                        for frame_mask in seg.cropped_mask[1:]:
                            combined_mask |= (frame_mask * 255).to(torch.uint8)

                        combined_mask = (combined_mask/255.0).to(torch.float32)
                        mask = utils.to_binary_mask(combined_mask, 0.1)
                    else:  # ndim == 2
                        mask = seg.cropped_mask

                    mask = tensor_gaussian_blur_mask(mask, feather) * (alpha/255)
                    x, y, *_ = seg.crop_region

                    # ensure same device
                    mask = mask.to(image_i.device)
                    ref_image = ref_image.to(image_i.device)

                    tensor_paste(image_i, ref_image, (x, y), mask)

            if result is None:
                result = image_i
            else:
                result = torch.concat((result, image_i), dim=0)

        if not args.highvram and not args.gpu_only:
            result = result.cpu()

        return (result, )

```
