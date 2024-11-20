---
tags:
- Mask
- MaskGeneration
---

# LayerMask: Image To Mask
## Documentation
- Class name: `LayerMask: ImageToMask`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

The `LayerMask: ImageToMask` node is designed to transform images into masks, focusing on extracting and processing mask layers from given images. It utilizes image processing techniques to create or modify mask layers, which can be used for further image manipulation or analysis.
## Input types
### Required
- **`image`**
    - The `image` parameter specifies the input image from which the mask will be generated. It is the primary source for mask creation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`channel`**
    - The `channel` parameter selects the specific color channel or image property to be used for mask generation, such as luminance or specific color channels, influencing the mask's characteristics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`black_point`**
    - The `black_point` parameter sets the lower threshold for mask generation, converting pixels darker than this value to black in the resulting mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`white_point`**
    - The `white_point` parameter sets the upper threshold for mask generation, converting pixels lighter than this value to white in the resulting mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`gray_point`**
    - The `gray_point` parameter adjusts the mid-tone levels in the mask, affecting the contrast and detail of the mask.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`invert_output_mask`**
    - The `invert_output_mask` parameter allows for the inversion of the generated mask, swapping its black and white areas.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`mask`**
    - The `mask` parameter, when provided, specifies an existing mask to be modified or combined with the newly generated mask from the image.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output `mask` is the transformed or newly created mask layer, which has been processed according to the specified parameters. It can be used for further image manipulation or analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageToMask:
    @classmethod
    def INPUT_TYPES(s):
        channel_list = ["L(LAB)", "A(Lab)", "B(Lab)",
                        "R(RGB)", "G(RGB)", "B(RGB)", "alpha",
                        "Y(YUV)", "U(YUV)", "V(YUV)",
                        "H(HSV)", "S(HSV", "V(HSV)"]
        return {
            "required": {
                "image": ("IMAGE", ),
                "channel": (channel_list,),
                "black_point": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "white_point": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "gray_point": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 9.99, "step": 0.01}),
                "invert_output_mask": ("BOOLEAN", {"default": False}),  # 反转mask
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "image_to_mask"
    CATEGORY = '😺dzNodes/LayerMask'

    def image_to_mask(self, image, channel,
                      black_point, white_point, gray_point,
                      invert_output_mask, mask=None
                      ):

        ret_masks = []
        l_images = []
        l_masks = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = []
            for m in mask:
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for i in range(len(l_images)):
            orig_image = l_images[i] if i < len(l_images) else l_images[-1]
            orig_image = tensor2pil(orig_image)
            orig_mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            mask = Image.new('L', orig_image.size, 'black')
            if channel == "L(LAB)":
                mask, _, _, _ = image_channel_split(orig_image, 'LAB')
            elif channel == "A(Lab)":
                _, mask, _, _ = image_channel_split(orig_image, 'LAB')
            elif channel == "B(Lab)":
                _, _, mask, _ = image_channel_split(orig_image, 'LAB')
            elif channel == "R(RGB)":
                mask, _, _, _ = image_channel_split(orig_image, 'RGB')
            elif channel == "G(RGB)":
                _, mask, _, _ = image_channel_split(orig_image, 'RGB')
            elif channel == "B(RGB)":
                _, _, mask, _ = image_channel_split(orig_image, 'RGB')
            elif channel == "alpha":
                _, _, _, mask = image_channel_split(orig_image, 'RGBA')
            elif channel == "Y(YUV)":
                mask, _, _, _ = image_channel_split(orig_image, 'YCbCr')
            elif channel == "U(YUV)":
                _, mask, _, _ = image_channel_split(orig_image, 'YCbCr')
            elif channel == "V(YUV)":
                _, _, mask, _ = image_channel_split(orig_image, 'YCbCr')
            elif channel == "H(HSV)":
                mask, _, _, _ = image_channel_split(orig_image, 'HSV')
            elif channel == "S(HSV)":
                _, mask, _, _ = image_channel_split(orig_image, 'HSV')
            elif channel == "V(HSV)":
                _, _, mask, _ = image_channel_split(orig_image, 'HSV')
            mask = normalize_gray(mask)
            mask = adjust_levels(mask, black_point, white_point, gray_point,
                          0, 255)
            if invert_output_mask:
                mask =  ImageChops.invert(mask)
            ret_mask = Image.new('L', mask.size, 'black')
            ret_mask.paste(mask, mask=orig_mask)

            ret_mask = image2mask(ret_mask)

            ret_masks.append(ret_mask)

        return (torch.cat(ret_masks, dim=0), )

```
