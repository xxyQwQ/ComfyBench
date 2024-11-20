---
tags:
- Segmentation
---

# 🔧 Apply CLIPSeg
## Documentation
- Class name: `ApplyCLIPSeg+`
- Category: `essentials/segmentation`
- Output node: `False`

This node applies CLIPSeg, a segmentation model, to an image based on a given prompt, threshold, and optional smoothing, dilation, and blurring parameters. It processes the image to segment it according to the semantics of the prompt, allowing for fine-tuned control over the segmentation output through post-processing steps.
## Input types
### Required
- **`clip_seg`**
    - A tuple containing the CLIPSeg processor and model, used for processing the image and prompt to generate segmentation masks.
    - Comfy dtype: `CLIP_SEG`
    - Python dtype: `Tuple[CLIPSegProcessor, CLIPSegForImageSegmentation]`
- **`image`**
    - The input image to be segmented. It is processed and segmented according to the given prompt and threshold.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`prompt`**
    - The text prompt that guides the segmentation process, influencing the areas of the image to be segmented.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`threshold`**
    - The threshold value for converting the model's output into a binary mask, determining the segmentation boundary.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`smooth`**
    - An optional parameter for applying Gaussian blur to smooth the edges of the segmented areas, enhancing the visual quality.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`dilate`**
    - An optional parameter for dilating the segmented areas, allowing for adjustments to the segmentation boundary.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur`**
    - An optional parameter for further blurring the segmented areas, useful for creating softer edges.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The segmented output as a tensor, where each pixel's value indicates whether it belongs to the segment defined by the prompt.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ApplyCLIPSeg:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_seg": ("CLIP_SEG",),
                "image": ("IMAGE",),
                "prompt": ("STRING", { "multiline": False, "default": "" }),
                "threshold": ("FLOAT", { "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05 }),
                "smooth": ("INT", { "default": 9, "min": 0, "max": 32, "step": 1 }),
                "dilate": ("INT", { "default": 0, "min": -32, "max": 32, "step": 1 }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 64, "step": 1 }),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/segmentation"

    def execute(self, image, clip_seg, prompt, threshold, smooth, dilate, blur):
        processor, model = clip_seg

        imagenp = image.mul(255).clamp(0, 255).byte().cpu().numpy()

        outputs = []
        for i in imagenp:
            inputs = processor(text=prompt, images=[i], return_tensors="pt")
            out = model(**inputs)
            out = out.logits.unsqueeze(1)
            out = torch.sigmoid(out[0][0])
            out = (out > threshold)
            outputs.append(out)

        del imagenp

        outputs = torch.stack(outputs, dim=0)

        if smooth > 0:
            if smooth % 2 == 0:
                smooth += 1
            outputs = T.functional.gaussian_blur(outputs, smooth)

        outputs = outputs.float()

        if dilate != 0:
            outputs = expand_mask(outputs, dilate, True)

        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            outputs = T.functional.gaussian_blur(outputs, blur)
        
        # resize to original size
        outputs = F.interpolate(outputs.unsqueeze(1), size=(image.shape[1], image.shape[2]), mode='bicubic').squeeze(1)

        return (outputs,)

```
