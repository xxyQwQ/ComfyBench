---
tags:
- Color
- ColorMatch
---

# Color Transfer
## Documentation
- Class name: `SAIColorTransfer`
- Category: `SALT/Image/Process`
- Output node: `False`

The SAIColorTransfer node is designed for applying color transfer techniques between images. It enables the modification of the color scheme of target images to match the color characteristics of source images, supporting various modes of color transfer.
## Input types
### Required
- **`target_images`**
    - Target images are the images whose color schemes are to be modified. They serve as the canvas for the color transfer process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`source_images`**
    - Source images provide the color scheme to be transferred to the target images. They act as the reference for the desired color characteristics.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mode`**
    - Specifies the color transfer mode to be used, such as 'pdf_regrain', 'mean_transfer', or 'lab_transfer', each offering a different approach to color adaptation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - The modified images with the color scheme of the source images applied to the target images.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SAIColorTransfer:
    def __init__(self):
        self.ct = ColorTransfer()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_images": ("IMAGE",),
                "source_images": ("IMAGE",),
                "mode": (["pdf_regrain", "mean_transfer", "lab_transfer"],)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "transfer"
    CATEGORY = "SALT/Image/Process"

    def transfer(self, target_images, source_images, mode):

        if target_images.shape[0] != source_images.shape[0]:
            repeat_factor = target_images.shape[0] // source_images.shape[0]
            source_images = source_images.repeat(repeat_factor, 1, 1, 1)

        results = []
        for target_image, source_image in zip(target_images, source_images):

            target_pil = tensor2pil(target_image)
            source_pil = tensor2pil(source_image)
            source_pil = source_pil.resize(target_pil.size)

            if mode == "pdf_regrain":
                res = pil2tensor(cv2pil(self.ct.pdf_transfer(img_arr_in=pil2cv(target_pil), img_arr_ref=pil2cv(source_pil), regrain=True)))
            elif mode == "mean_transfer":
                res = pil2tensor(cv2pil(self.ct.mean_std_transfer(img_arr_in=pil2cv(target_pil), img_arr_ref=pil2cv(source_pil))))
            elif mode == "lab_transfer":
                res = pil2tensor(cv2pil(self.ct.lab_transfer(img_arr_in=pil2cv(target_pil), img_arr_ref=pil2cv(source_pil))))
            else:
                errmsg = f"Invalid mode `{mode}` selected for {self.__class__.__name__}"
                logger.warning(errmsg)
                res = target_image

            results.append(res)

        results = torch.cat(results, dim=0)

        return (results, )

```
