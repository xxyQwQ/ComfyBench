---
tags:
- DepthMap
- DepthMapEstimation
- Image
---

# MeshGraphormer Hand Refiner
## Documentation
- Class name: `MeshGraphormer-DepthMapPreprocessor`
- Category: `ControlNet Preprocessors/Normal and Depth Estimators`
- Output node: `False`

This node is designed to preprocess images for depth map generation, specifically tailored for hand gestures. It utilizes a MeshGraphormer model to refine depth maps and masks of hands within images, enhancing the accuracy of depth perception for each detected hand region.
## Input types
### Required
- **`image`**
    - The input image to be processed for hand gesture depth map generation. It serves as the primary data for detecting hand landmarks and generating corresponding depth maps and masks.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
### Optional
- **`mask_bbox_padding`**
    - Specifies the padding around the bounding box of detected hands, affecting the area considered for depth map generation. It helps in adjusting the focus area around the hands.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`resolution`**
    - The resolution to which the input images are resized before processing. This parameter directly impacts the model's performance and the quality of the output depth maps.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`mask_type`**
    - Determines the type of mask to be generated, such as based on depth values or tight bounding boxes, influencing how hand regions are isolated from the background.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`mask_expand`**
    - Defines the expansion or contraction of the mask boundaries, allowing for finer control over the size of the hand region to be processed.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rand_seed`**
    - A seed value for random number generation, ensuring reproducibility of the depth maps and masks across multiple runs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`detect_thr`**
    - The detection threshold for the MeshGraphormer model, determining the sensitivity of hand detection within the images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`presence_thr`**
    - The presence threshold for the MeshGraphormer model, affecting the likelihood of a hand's presence being recognized in the processed area.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The refined depth maps for each hand region detected in the input images. These maps provide detailed depth information, crucial for subsequent processing steps.
    - Python dtype: `torch.Tensor`
- **`INPAINTING_MASK`**
    - Comfy dtype: `MASK`
    - Binary masks corresponding to the hand regions within the input images. These masks are essential for isolating hand gestures from the background.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)



## Source code
```python
class Mesh_Graphormer_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        types = create_node_input_types(mask_bbox_padding=("INT", {"default": 30, "min": 0, "max": 100}))
        types["optional"].update(
            {
                "mask_type": (["based_on_depth", "tight_bboxes", "original"], {"default": "based_on_depth"}),
                "mask_expand": ("INT", {"default": 5, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "rand_seed": ("INT", {"default": 88, "min": 0, "max": 0xffffffffffffffff}),
                "detect_thr": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01}),
                "presence_thr": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01}),
            }
        )
        return types

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "INPAINTING_MASK")
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, mask_bbox_padding=30, mask_type="based_on_depth", mask_expand=5, resolution=512, rand_seed=88, detect_thr=0.6, presence_thr=0.6, **kwargs):
        install_deps()
        from controlnet_aux.mesh_graphormer import MeshGraphormerDetector
        model = kwargs["model"] if "model" in kwargs \
            else MeshGraphormerDetector.from_pretrained(detect_thr=detect_thr, presence_thr=presence_thr).to(model_management.get_torch_device())
        
        depth_map_list = []
        mask_list = []
        for single_image in image:
            np_image = np.asarray(single_image.cpu() * 255., dtype=np.uint8)
            depth_map, mask, info = model(np_image, output_type="np", detect_resolution=resolution, mask_bbox_padding=mask_bbox_padding, seed=rand_seed)
            if mask_type == "based_on_depth":
                H, W = mask.shape[:2]
                mask = cv2.resize(depth_map.copy(), (W, H))
                mask[mask > 0] = 255

            elif mask_type == "tight_bboxes":
                mask = np.zeros_like(mask)
                hand_bboxes = info["abs_boxes"]
                for hand_bbox in hand_bboxes: 
                    x_min, x_max, y_min, y_max = hand_bbox
                    mask[y_min:y_max+1, x_min:x_max+1, :] = 255 #HWC

            mask = mask[:, :, :1]
            depth_map_list.append(torch.from_numpy(depth_map.astype(np.float32) / 255.0))
            mask_list.append(torch.from_numpy(mask.astype(np.float32) / 255.0))
        depth_maps, masks = torch.stack(depth_map_list, dim=0), rearrange(torch.stack(mask_list, dim=0), "n h w 1 -> n 1 h w")
        return depth_maps, expand_mask(masks, mask_expand, tapered_corners=True)

```
