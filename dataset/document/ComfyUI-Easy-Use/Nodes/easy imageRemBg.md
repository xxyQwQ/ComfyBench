---
tags:
- BackgroundRemoval
- Image
---

# Image Remove Bg
## Documentation
- Class name: `easy imageRemBg`
- Category: `EasyUse/Image`
- Output node: `True`

This node specializes in removing backgrounds from images, offering a streamlined approach for image processing tasks that require isolating subjects from their backgrounds. It leverages specific removal modes to adapt the background removal process to various requirements, enhancing flexibility and utility in image manipulation workflows.
## Input types
### Required
- **`images`**
    - Specifies the images from which backgrounds are to be removed. This parameter is crucial for defining the input images that the node will process, directly influencing the outcome of the background removal.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`rem_mode`**
    - Determines the removal mode to be used, which can affect the method and quality of the background removal. This allows for customization of the background removal process based on specific needs or preferences.
    - Comfy dtype: `['RMBG-1.4']`
    - Python dtype: `str`
- **`image_output`**
    - Defines how the output image should be handled, whether it should be hidden, previewed, saved, or both hidden and saved. This parameter allows for flexible management of the output image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`save_prefix`**
    - Specifies the prefix for saved images, providing a way to organize and identify output files easily.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image with the background removed. This highlights the node's ability to isolate the subject from its background, providing a clean image ready for further processing or use.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The mask generated during the background removal process. This can be used for further image processing tasks, such as compositing or additional manipulations.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class imageRemBg:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "images": ("IMAGE",),
        "rem_mode": (("RMBG-1.4",),),
        "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
        "save_prefix": ("STRING", {"default": "ComfyUI"}),
      },
      "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
    }

  RETURN_TYPES = ("IMAGE", "MASK")
  RETURN_NAMES = ("image", "mask")
  FUNCTION = "remove"
  OUTPUT_NODE = True

  CATEGORY = "EasyUse/Image"

  def remove(self, rem_mode, images, image_output, save_prefix, prompt=None, extra_pnginfo=None):
    if rem_mode == "RMBG-1.4":
      # load model
      model_url = REMBG_MODELS[rem_mode]['model_url']
      suffix = model_url.split(".")[-1]
      model_path = get_local_filepath(model_url, REMBG_DIR, rem_mode+'.'+suffix)

      net = BriaRMBG()
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      net.load_state_dict(torch.load(model_path, map_location=device))
      net.to(device)
      net.eval()
      # prepare input
      model_input_size = [1024, 1024]
      new_images = list()
      masks = list()
      for image in images:
        orig_im = tensor2pil(image)
        w, h = orig_im.size
        image = preprocess_image(orig_im, model_input_size).to(device)
        # inference
        result = net(image)
        result_image = postprocess_image(result[0][0], (h, w))
        mask_im = Image.fromarray(result_image)
        new_im = Image.new("RGBA", mask_im.size, (0,0,0,0))
        new_im.paste(orig_im, mask=mask_im)

        new_images.append(pil2tensor(new_im))
        masks.append(pil2tensor(mask_im))

      new_images = torch.cat(new_images, dim=0)
      masks = torch.cat(masks, dim=0)


      results = easySave(new_images, save_prefix, image_output, prompt, extra_pnginfo)

      if image_output in ("Hide", "Hide/Save"):
        return {"ui": {},
                "result": (new_images, masks)}

      return {"ui": {"images": results},
              "result": (new_images, masks)}

    else:
      return (None, None)

```
