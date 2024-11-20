---
tags:
- DepthMap
- DepthMapEstimation
- Image
---

# MeshGraphormer Hand Refiner With External Detector
## Documentation
- Class name: `MeshGraphormer+ImpactDetector-DepthMapPreprocessor`
- Category: `ControlNet Preprocessors/Normal and Depth Estimators`
- Output node: `False`

This node integrates an external impact detector with the MeshGraphormer model to preprocess images for depth map and mask generation. It enhances the depth map preprocessing by incorporating detection of specific features (e.g., hands) in the images, refining the depth map and mask outputs based on detected regions.
## Input types
### Required
- **`image`**
    - The input image to be processed for depth map and mask generation, serving as the primary data for the node's operations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`bbox_detector`**
    - Specifies the bounding box detector to be used for identifying regions of interest within the images, crucial for the subsequent depth map and mask generation.
    - Comfy dtype: `BBOX_DETECTOR`
    - Python dtype: `BBOX_DETECTOR`
### Optional
- **`bbox_threshold`**
    - The threshold value for bounding box detection, controlling the sensitivity of the external impact detector in identifying regions of interest.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`bbox_dilation`**
    - The dilation factor applied to bounding boxes, adjusting the area around detected features for depth map generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`bbox_crop_factor`**
    - The factor by which the bounding box is cropped, focusing the depth map generation on more precise areas of interest.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`drop_size`**
    - The minimum size of detected features to be considered, filtering out smaller detections to focus on significant regions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`mask_bbox_padding`**
    - The padding added around the bounding box when generating masks, allowing for more flexible mask creation around detected features.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`mask_type`**
    - Defines the method used for mask generation, whether based on depth information, tight bounding boxes, or the original detection method.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`mask_expand`**
    - The expansion factor for masks, adjusting the area covered by the mask beyond the detected features.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rand_seed`**
    - The random seed used for deterministic operations within the model, ensuring consistent results across runs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`resolution`**
    - Specifies the resolution at which the depth map and mask are generated, impacting the detail and quality of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The generated depth maps corresponding to the input images, refined based on detected features and regions of interest.
    - Python dtype: `torch.Tensor`
- **`INPAINTING_MASK`**
    - Comfy dtype: `MASK`
    - The masks indicating the areas of interest within the images, generated based on the depth map preprocessing and feature detection.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class Mesh_Graphormer_With_ImpactDetector_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        types = create_node_input_types(
            # Impact pack
            bbox_threshold=("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            bbox_dilation=("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
            bbox_crop_factor=("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
            drop_size=("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
            # Mesh Graphormer
            mask_bbox_padding=("INT", {"default": 30, "min": 0, "max": 100}),
            mask_type=(["based_on_depth", "tight_bboxes", "original"], {"default": "based_on_depth"}),
            mask_expand=("INT", {"default": 5, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
            rand_seed=("INT", {"default": 88, "min": 0, "max": 0xffffffffffffffff}),
        )
        types["required"]["bbox_detector"] = ("BBOX_DETECTOR", )
        return types
     
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "INPAINTING_MASK")
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, bbox_detector, bbox_threshold=0.5, bbox_dilation=10, bbox_crop_factor=3.0, drop_size=10, resolution=512, **mesh_graphormer_kwargs):
        install_deps()
        from controlnet_aux.mesh_graphormer import MeshGraphormerDetector
        mesh_graphormer_node = Mesh_Graphormer_Depth_Map_Preprocessor()
        model = MeshGraphormerDetector.from_pretrained(detect_thr=0.6, presence_thr=0.6).to(model_management.get_torch_device())
        mesh_graphormer_kwargs["model"] = model

        frames = image
        depth_maps, masks = [], []
        for idx in range(len(frames)):
            frame = frames[idx:idx+1,...] #Impact Pack's BBOX_DETECTOR only supports single batch image
            bbox_detector.setAux('face') # make default prompt as 'face' if empty prompt for CLIPSeg
            _, segs = bbox_detector.detect(frame, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)
            bbox_detector.setAux(None)

            n, h, w, _ = frame.shape
            depth_map, mask = torch.zeros_like(frame), torch.zeros(n, 1, h, w)
            for i, seg in enumerate(segs):
                x1, y1, x2, y2 = seg.crop_region
                cropped_image = frame[:, y1:y2, x1:x2, :]  # Never use seg.cropped_image to handle overlapping area
                mesh_graphormer_kwargs["resolution"] = 0 #Disable resizing
                sub_depth_map, sub_mask = mesh_graphormer_node.execute(cropped_image, **mesh_graphormer_kwargs)
                depth_map[:, y1:y2, x1:x2, :] = sub_depth_map
                mask[:, :, y1:y2, x1:x2] = sub_mask
            
            depth_maps.append(depth_map)
            masks.append(mask)
            
        return (torch.cat(depth_maps), torch.cat(masks))

```
