---
tags:
- Mask
---

# LayerMask: Mediapipe Facial Segment
## Documentation
- Class name: `LayerMask: MediapipeFacialSegment`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

This node leverages the Mediapipe library to perform facial feature segmentation on images. It extracts specific facial features based on user input, such as eyes, eyebrows, lips, and teeth, and generates corresponding masks for each feature.
## Input types
### Required
- **`image`**
    - The input image on which facial feature segmentation is to be performed. It is essential for identifying and extracting the facial features.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`left_eye`**
    - A boolean indicating whether to include the left eye in the facial feature segmentation. This affects the output mask by including or excluding the left eye region.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`left_eyebrow`**
    - A boolean indicating whether to include the left eyebrow in the segmentation. This choice influences the resulting mask by determining if the left eyebrow area is segmented.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`right_eye`**
    - A boolean parameter that controls whether the right eye's region is included in the segmentation output. It affects the mask generation process by including or excluding the right eye.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`right_eyebrow`**
    - Determines whether the right eyebrow is included in the facial feature segmentation. This parameter influences the mask creation by specifying if the right eyebrow area should be segmented.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`lips`**
    - Indicates whether the lips should be segmented in the facial feature extraction. This parameter affects the output by including or excluding the lips in the generated mask.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`tooth`**
    - Controls whether the teeth are included in the segmentation. This affects the final mask by determining if the teeth area is segmented.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The original image with facial features segmented based on the input parameters.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - A mask image highlighting the segmented facial features. Each feature segmented based on the input parameters is represented in the mask.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FacialFeatureSegment:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE",),  #
                "left_eye": ("BOOLEAN", {"default": True}),
                "left_eyebrow": ("BOOLEAN", {"default": True}),
                "right_eye": ("BOOLEAN", {"default": True}),
                "right_eyebrow": ("BOOLEAN", {"default": True}),
                "lips": ("BOOLEAN", {"default": True}),
                "tooth": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = 'facial_feature_segment'
    CATEGORY = '😺dzNodes/LayerMask'

    def facial_feature_segment(self, image,
                              left_eye, left_eyebrow, right_eye, right_eyebrow, lips, tooth
                  ):

        # 定义面部特征索引
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
        left_eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        right_eyebrow_indices = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
        # upper_lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
        # lower_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
        tooth_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
        lips_indices = [61, 76, 62, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178,
                         88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
                         61]

        ret_images = []
        ret_masks = []
        scale_factor = 4

        for i in image:
            face_image = tensor2pil(i.unsqueeze(0)).convert('RGB')
            width, height = face_image.size
            width *= scale_factor
            height *= scale_factor
            cv2_image = pil2cv2(face_image)
            mp_face_mesh = mp.solutions.face_mesh
            fase_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
            results = fase_mesh.process(cv2_image)
            mask = np.zeros((height, width), dtype=np.uint8)


            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 绘制各个面部特征
                    if left_eye:
                        draw_feature(left_eye_indices, mask, face_landmarks, width, height)
                    if right_eye:
                        draw_feature(right_eye_indices, mask, face_landmarks, width, height)
                    if left_eyebrow:
                        draw_feature(left_eyebrow_indices, mask, face_landmarks, width, height)
                    if right_eyebrow:
                        draw_feature(right_eyebrow_indices, mask, face_landmarks, width, height)
                    if lips:
                        draw_feature(lips_indices, mask, face_landmarks, width, height)
                    if tooth:
                        draw_feature(tooth_indices, mask, face_landmarks, width, height)

            mask = cv22pil(mask).convert('L')
            mask = gaussian_blur(mask, 2)
            mask = mask.resize(face_image.size, Image.BILINEAR)
            ret_images.append(pil2tensor(RGB2RGBA(face_image, mask)))
            ret_masks.append(image2mask(mask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

```
