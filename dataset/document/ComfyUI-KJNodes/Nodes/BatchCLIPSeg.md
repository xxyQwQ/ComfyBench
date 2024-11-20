---
tags:
- Segmentation
---

# Batch CLIPSeg
## Documentation
- Class name: `BatchCLIPSeg`
- Category: `KJNodes/masking`
- Output node: `False`

This node is designed for batch processing of images using the CLIPSeg model for image segmentation. It leverages the CLIPSegForImageSegmentation and CLIPSegProcessor from the transformers library to perform semantic segmentation on a collection of images, adapting the model and processor to the specific hardware configuration (CPU or GPU) and data type for efficient execution.
## Input types
### Required
- **`images`**
    - The input images to be processed for segmentation. This parameter is crucial for the node's operation as it directly affects the segmentation results.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`text`**
    - The text prompt used to guide the segmentation process, influencing the areas of the image that will be segmented.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`threshold`**
    - A threshold value for determining the segmentation cut-off, affecting the sensitivity of the segmentation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`binary_mask`**
    - A boolean flag indicating whether the output mask should be binary, affecting the format of the segmentation result.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`combine_mask`**
    - A boolean flag indicating whether to combine the output masks for batch processing, affecting the structure of the output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`use_cuda`**
    - A flag indicating whether to use CUDA for processing. This affects the execution speed and efficiency, especially for large batches of images.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`blur_sigma`**
    - The sigma value for Gaussian blur to apply to the output mask, affecting the smoothness of the mask edges.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`opt_model`**
    - An optional pre-loaded model and processor to use for segmentation, allowing for flexibility in model choice and potential reuse of resources.
    - Comfy dtype: `CLIPSEGMODEL`
    - Python dtype: `dict`
- **`prev_mask`**
    - An optional previous mask to be combined with the current segmentation, allowing for iterative segmentation refinement.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`image_bg_level`**
    - The background level for the image, affecting the contrast and visibility of the segmentation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`invert`**
    - A boolean flag indicating whether to invert the output mask, affecting the segmentation's focus area.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`Mask`**
    - Comfy dtype: `MASK`
    - The output mask generated from the segmentation process, providing a binary or probabilistic map of the segmented areas.
    - Python dtype: `torch.Tensor`
- **`Image`**
    - Comfy dtype: `IMAGE`
    - The original image overlaid with the segmentation mask, visually representing the segmentation results.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BatchCLIPSeg:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
       
        return {"required":
                    {
                        "images": ("IMAGE",),
                        "text": ("STRING", {"multiline": False}),
                        "threshold": ("FLOAT", {"default": 0.5,"min": 0.0, "max": 10.0, "step": 0.001}),
                        "binary_mask": ("BOOLEAN", {"default": True}),
                        "combine_mask": ("BOOLEAN", {"default": False}),
                        "use_cuda": ("BOOLEAN", {"default": True}),
                     },
                     "optional":
                    {
                        "blur_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                        "opt_model": ("CLIPSEGMODEL", ),
                        "prev_mask": ("MASK", {"default": None}),
                        "image_bg_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "invert": ("BOOLEAN", {"default": False}),
                    }
                }

    CATEGORY = "KJNodes/masking"
    RETURN_TYPES = ("MASK", "IMAGE", )
    RETURN_NAMES = ("Mask", "Image", )
    FUNCTION = "segment_image"
    DESCRIPTION = """
Segments an image or batch of images using CLIPSeg.
"""

    def segment_image(self, images, text, threshold, binary_mask, combine_mask, use_cuda, blur_sigma=0.0, opt_model=None, prev_mask=None, invert= False, image_bg_level=0.5):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        import torchvision.transforms as transforms
        offload_device = model_management.unet_offload_device()
        device = model_management.get_torch_device()
        if not use_cuda:
            device = torch.device("cpu")
        dtype = model_management.unet_dtype()

        if opt_model is None:
            checkpoint_path = os.path.join(folder_paths.models_dir,'clip_seg', 'clipseg-rd64-refined-fp16')
            if not hasattr(self, "model"):
                try:
                    if not os.path.exists(checkpoint_path):
                        from huggingface_hub import snapshot_download
                        snapshot_download(repo_id="Kijai/clipseg-rd64-refined-fp16", local_dir=checkpoint_path, local_dir_use_symlinks=False)
                    self.model = CLIPSegForImageSegmentation.from_pretrained(checkpoint_path)
                except:
                    checkpoint_path = "CIDAS/clipseg-rd64-refined"
                    self.model = CLIPSegForImageSegmentation.from_pretrained(checkpoint_path)
            processor = CLIPSegProcessor.from_pretrained(checkpoint_path)

        else:
            self.model = opt_model['model']
            processor = opt_model['processor']

        self.model.to(dtype).to(device)

        B, H, W, C = images.shape
        images = images.to(device)
        
        autocast_condition = (dtype != torch.float32) and not model_management.is_device_mps(device)
        with torch.autocast(model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():

            PIL_images = [Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)) for image in images ]
            prompt = [text] * len(images)
            input_prc = processor(text=prompt, images=PIL_images, return_tensors="pt")

            for key in input_prc:
                input_prc[key] = input_prc[key].to(device)
            outputs = self.model(**input_prc)

        mask_tensor = torch.sigmoid(outputs.logits)
        mask_tensor = (mask_tensor - mask_tensor.min()) / (mask_tensor.max() - mask_tensor.min())
        mask_tensor = torch.where(mask_tensor > (threshold), mask_tensor, torch.tensor(0, dtype=torch.float))
        print(mask_tensor.shape)
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(1), size=(H, W), mode='nearest')
        mask_tensor = mask_tensor.squeeze(1)

        self.model.to(offload_device)
        
        if binary_mask:
            mask_tensor = (mask_tensor > 0).float()
        if blur_sigma > 0:
            kernel_size = int(6 * int(blur_sigma) + 1) 
            blur = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(blur_sigma, blur_sigma))
            mask_tensor = blur(mask_tensor)

        if combine_mask:
            mask_tensor = torch.max(mask_tensor, dim=0)[0]
            mask_tensor = mask_tensor.unsqueeze(0).repeat(len(images),1,1)

        del outputs
        model_management.soft_empty_cache()

        if prev_mask is not None:
            if prev_mask.shape != mask_tensor.shape:
                prev_mask = F.interpolate(prev_mask.unsqueeze(1), size=(H, W), mode='nearest')
            mask_tensor = mask_tensor + prev_mask.to(device)
            torch.clamp(mask_tensor, min=0.0, max=1.0)

        if invert:
            mask_tensor = 1 - mask_tensor

        image_tensor = images * mask_tensor.unsqueeze(-1) + (1 - mask_tensor.unsqueeze(-1)) * image_bg_level
        image_tensor = torch.clamp(image_tensor, min=0.0, max=1.0).cpu().float()

        mask_tensor = mask_tensor.cpu().float()
    
        return mask_tensor, image_tensor, 

```
