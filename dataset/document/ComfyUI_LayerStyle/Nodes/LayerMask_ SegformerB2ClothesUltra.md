---
tags:
- Segmentation
---

# LayerMask: Segformer B2 Clothes Ultra
## Documentation
- Class name: `LayerMask: SegformerB2ClothesUltra`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

This node specializes in segmenting clothing from images using the Segformer B2 model. It processes images to identify and isolate clothing items, leveraging advanced semantic segmentation techniques to achieve high precision and detail in the segmentation results.
## Input types
### Required
- **`image`**
    - The input image, which the node processes to segment clothing. It's crucial for the segmentation task, as the model applies semantic segmentation techniques to this input to isolate clothing items.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`face`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`hair`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`hat`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`sunglass`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`left_arm`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`right_arm`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`left_leg`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`right_leg`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`upper_clothes`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`skirt`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`pants`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`dress`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`belt`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`shoe`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`bag`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`scarf`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`detail_method`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`detail_erode`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`detail_dilate`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`black_point`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`white_point`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`process_detail`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`device`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`max_megapixels`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The segmented image with clothing items isolated, returned as a tensor. It represents the visual result of the segmentation process.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The segmentation mask, indicating the areas of the image identified as clothing. It provides a binary or multi-class mask highlighting the segmented clothing areas.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class Segformer_B2_Clothes:

    def __init__(self):
        pass
    
    # Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt",
    # 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face",
    # 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"
    
    @classmethod
    def INPUT_TYPES(cls):
        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda','cpu']
        return {"required":
                {     
                "image":("IMAGE",),
                "face": ("BOOLEAN", {"default": False}),
                "hair": ("BOOLEAN", {"default": False}),
                "hat": ("BOOLEAN", {"default": False}),
                "sunglass": ("BOOLEAN", {"default": False}),
                "left_arm": ("BOOLEAN", {"default": False}),
                "right_arm": ("BOOLEAN", {"default": False}),
                "left_leg": ("BOOLEAN", {"default": False}),
                "right_leg": ("BOOLEAN", {"default": False}),
                "upper_clothes": ("BOOLEAN", {"default": False}),
                "skirt": ("BOOLEAN", {"default": False}),
                "pants": ("BOOLEAN", {"default": False}),
                "dress": ("BOOLEAN", {"default": False}),
                "belt": ("BOOLEAN", {"default": False}),
                "shoe": ("BOOLEAN", {"default": False}),
                "bag": ("BOOLEAN", {"default": False}),
                "scarf": ("BOOLEAN", {"default": False}),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 12, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
                }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "segformer_ultra"
    CATEGORY = '😺dzNodes/LayerMask'

    def segformer_ultra(self, image,
                        face, hat, hair, sunglass, upper_clothes, skirt, pants, dress, belt, shoe,
                        left_leg, right_leg, left_arm, right_arm, bag, scarf, detail_method,
                        detail_erode, detail_dilate, black_point, white_point, process_detail, device, max_megapixels,
                        ):

        ret_images = []
        ret_masks = []

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        for i in image:
            pred_seg, cloth = get_segmentation(i)
            i = torch.unsqueeze(i, 0)
            i = pil2tensor(tensor2pil(i).convert('RGB'))
            orig_image = tensor2pil(i).convert('RGB')

            labels_to_keep = [0]
            if not hat:
                labels_to_keep.append(1)
            if not hair:
                labels_to_keep.append(2)
            if not sunglass:
                labels_to_keep.append(3)
            if not upper_clothes:
                labels_to_keep.append(4)
            if not skirt:
                labels_to_keep.append(5)
            if not pants:
                labels_to_keep.append(6)
            if not dress:
                labels_to_keep.append(7)
            if not belt:
                labels_to_keep.append(8)
            if not shoe:
                labels_to_keep.append(9)
                labels_to_keep.append(10)
            if not face:
                labels_to_keep.append(11)
            if not left_leg:
                labels_to_keep.append(12)
            if not right_leg:
                labels_to_keep.append(13)
            if not left_arm:
                labels_to_keep.append(14)
            if not right_arm:
                labels_to_keep.append(15)
            if not bag:
                labels_to_keep.append(16)
            if not scarf:
                labels_to_keep.append(17)

            mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)

            # 创建agnostic-mask图像
            mask_image = Image.fromarray((1 - mask) * 255)
            mask_image = mask_image.convert("L")
            _mask = pil2tensor(mask_image)

            detail_range = detail_erode + detail_dilate
            if process_detail:
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(i, _mask, detail_range // 6 + 1)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range // 8 + 1, black_point, white_point))
                else:
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
            else:
                _mask = mask2image(_mask)

            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

```
