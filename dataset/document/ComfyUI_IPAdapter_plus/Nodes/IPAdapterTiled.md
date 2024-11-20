---
tags:
- IPAdapter
---

# IPAdapter Tiled
## Documentation
- Class name: `IPAdapterTiled`
- Category: `ipadapter/tiled`
- Output node: `False`

The IPAdapterTiled node is designed to apply image processing adaptations in a tiled manner, allowing for detailed and scalable modifications to images. It leverages various image processing techniques and parameters to enhance, modify, or transform images based on the provided inputs and configurations.
## Input types
### Required
- **`model`**
    - Specifies the model to be used for image processing, serving as the core component for adaptations.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`ipadapter`**
    - Defines the IPAdapter configuration to be applied, dictating the specific image processing techniques and parameters.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `str`
- **`image`**
    - The input image to be processed, serving as the primary subject for the adaptations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`weight`**
    - A floating-point value that adjusts the intensity of the applied adaptations, allowing for fine-tuned control over the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_type`**
    - Determines the method of weighting the adaptations, influencing how different aspects of the image are emphasized or blended.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`combine_embeds`**
    - Determines how embeddings are combined, offering options like concatenation to influence the final image adaptation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_at`**
    - A floating-point value indicating the starting point of the adaptations, enabling phased or gradual application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - A floating-point value marking the end point of the adaptations, allowing for precise control over the extent of modifications.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sharpening`**
    - A floating-point value that controls the level of sharpening applied to the image, enhancing detail and clarity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`embeds_scaling`**
    - Specifies the scaling method for embeddings, affecting the adaptation's influence on different image features.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list`
### Optional
- **`image_negative`**
    - An optional input image representing negative aspects to be minimized or avoided in the adaptations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`attn_mask`**
    - An optional mask that directs attention to specific areas of the image, focusing adaptations where needed.
    - Comfy dtype: `MASK`
    - Python dtype: `str`
- **`clip_vision`**
    - An optional parameter that integrates CLIP vision models for guided image adaptations, enhancing relevance and coherence.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `str`
## Output types
- **`MODEL`**
    - Comfy dtype: `MODEL`
    - The processed model after image adaptations have been applied.
    - Python dtype: `str`
- **`tiles`**
    - Comfy dtype: `IMAGE`
    - The resulting image tiles after processing, showcasing the segmented adaptations.
    - Python dtype: `list`
- **`masks`**
    - Comfy dtype: `MASK`
    - The masks applied to each tile, indicating areas of focus or modification.
    - Python dtype: `list`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterTiled:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "sharpening": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "MASK", )
    RETURN_NAMES = ("MODEL", "tiles", "masks", )
    FUNCTION = "apply_tiled"
    CATEGORY = "ipadapter/tiled"

    def apply_tiled(self, model, ipadapter, image, weight, weight_type, start_at, end_at, sharpening, combine_embeds="concat", image_negative=None, attn_mask=None, clip_vision=None, embeds_scaling='V only', encode_batch_size=0):
        # 1. Select the models
        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision

        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")

        del ipadapter

        # 2. Extract the tiles
        tile_size = 256     # I'm using 256 instead of 224 as it is more likely divisible by the latent size, it will be downscaled to 224 by the clip vision encoder
        _, oh, ow, _ = image.shape
        if attn_mask is None:
            attn_mask = torch.ones([1, oh, ow], dtype=image.dtype, device=image.device)

        image = image.permute([0,3,1,2])
        attn_mask = attn_mask.unsqueeze(1)
        # the mask should have the same proportions as the reference image and the latent
        attn_mask = T.Resize((oh, ow), interpolation=T.InterpolationMode.BICUBIC, antialias=True)(attn_mask)

        # if the image is almost a square, we crop it to a square
        if oh / ow > 0.75 and oh / ow < 1.33:
            # crop the image to a square
            image = T.CenterCrop(min(oh, ow))(image)
            resize = (tile_size*2, tile_size*2)

            attn_mask = T.CenterCrop(min(oh, ow))(attn_mask)
        # otherwise resize the smallest side and the other proportionally
        else:
            resize = (int(tile_size * ow / oh), tile_size) if oh < ow else (tile_size, int(tile_size * oh / ow))

         # using PIL for better results
        imgs = []
        for img in image:
            img = T.ToPILImage()(img)
            img = img.resize(resize, resample=Image.Resampling['LANCZOS'])
            imgs.append(T.ToTensor()(img))
        image = torch.stack(imgs)
        del imgs, img

        # we don't need a high quality resize for the mask
        attn_mask = T.Resize(resize[::-1], interpolation=T.InterpolationMode.BICUBIC, antialias=True)(attn_mask)

        # we allow a maximum of 4 tiles
        if oh / ow > 4 or oh / ow < 0.25:
            crop = (tile_size, tile_size*4) if oh < ow else (tile_size*4, tile_size)
            image = T.CenterCrop(crop)(image)
            attn_mask = T.CenterCrop(crop)(attn_mask)

        attn_mask = attn_mask.squeeze(1)

        if sharpening > 0:
            image = contrast_adaptive_sharpening(image, sharpening)

        image = image.permute([0,2,3,1])

        _, oh, ow, _ = image.shape

        # find the number of tiles for each side
        tiles_x = math.ceil(ow / tile_size)
        tiles_y = math.ceil(oh / tile_size)
        overlap_x = max(0, (tiles_x * tile_size - ow) / (tiles_x - 1 if tiles_x > 1 else 1))
        overlap_y = max(0, (tiles_y * tile_size - oh) / (tiles_y - 1 if tiles_y > 1 else 1))

        base_mask = torch.zeros([attn_mask.shape[0], oh, ow], dtype=image.dtype, device=image.device)

        # extract all the tiles from the image and create the masks
        tiles = []
        masks = []
        for y in range(tiles_y):
            for x in range(tiles_x):
                start_x = int(x * (tile_size - overlap_x))
                start_y = int(y * (tile_size - overlap_y))
                tiles.append(image[:, start_y:start_y+tile_size, start_x:start_x+tile_size, :])
                mask = base_mask.clone()
                mask[:, start_y:start_y+tile_size, start_x:start_x+tile_size] = attn_mask[:, start_y:start_y+tile_size, start_x:start_x+tile_size]
                masks.append(mask)
        del mask

        # 3. Apply the ipadapter to each group of tiles
        model = model.clone()
        for i in range(len(tiles)):
            ipa_args = {
                "image": tiles[i],
                "image_negative": image_negative,
                "weight": weight,
                "weight_type": weight_type,
                "combine_embeds": combine_embeds,
                "start_at": start_at,
                "end_at": end_at,
                "attn_mask": masks[i],
                "unfold_batch": self.unfold_batch,
                "embeds_scaling": embeds_scaling,
                "encode_batch_size": encode_batch_size,
            }
            # apply the ipadapter to the model without cloning it
            model, _ = ipadapter_execute(model, ipadapter_model, clip_vision, **ipa_args)

        return (model, torch.cat(tiles), torch.cat(masks), )

```
