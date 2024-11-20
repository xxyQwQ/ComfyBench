# Upper Body Tracking From PoseKps (InstanceDiffusion)
## Documentation
- Class name: `UpperBodyTrackingFromPoseKps`
- Category: `ControlNet Preprocessors/Pose Keypoint Postprocess`
- Output node: `False`

This node is designed to process pose keypoints for upper body tracking, extracting and formatting specific body parts such as the torso, arms, and face from pose detection results. It enhances the keypoints data by appending additional points for completeness and accuracy, particularly for facial features, and compiles the results into structured pose results for further analysis or visualization.
## Input types
### Required
- **`pose_kps`**
    - The input pose keypoints, which are essential for identifying and tracking the upper body parts in images or video frames. This data includes body, hands, and face keypoints along with their scores.
    - Comfy dtype: `POSE_KEYPOINT`
    - Python dtype: `List[ndarray]`
- **`id_include`**
    - An optional string to include specific IDs in the tracking process, allowing for targeted analysis of pose keypoints.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`Head_width_height`**
    - Specifies the width and height for the head part, used in calculating dimensions for tracking.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`Neck_width_height`**
    - Specifies the width and height for the neck part, used in calculating dimensions for tracking.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`Shoulder_width_height`**
    - Specifies the width and height for the shoulder part, used in calculating dimensions for tracking.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`Torso_width_height`**
    - Specifies the width and height for the torso part, used in calculating dimensions for tracking.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`RArm_width_height`**
    - Specifies the width and height for the right arm part, used in calculating dimensions for tracking.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`RForearm_width_height`**
    - Specifies the width and height for the right forearm part, used in calculating dimensions for tracking.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`LArm_width_height`**
    - Specifies the width and height for the left arm part, used in calculating dimensions for tracking.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`LForearm_width_height`**
    - Specifies the width and height for the left forearm part, used in calculating dimensions for tracking.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`tracking`**
    - Comfy dtype: `TRACKING`
    - Structured data representing the tracking of upper body parts, including detailed information about body keypoints, hands, and face, along with their respective scores.
    - Python dtype: `str`
- **`prompt`**
    - Comfy dtype: `STRING`
    - A textual prompt or summary generated based on the tracking results, providing insights or conclusions from the analysis.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class UpperBodyTrackingFromPoseKps:
    PART_NAMES = ["Head", "Neck", "Shoulder", "Torso", "RArm", "RForearm", "LArm", "LForearm"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_kps": ("POSE_KEYPOINT",),
                "id_include": ("STRING", {"default": '', "multiline": False}),
                **{part_name + "_width_height": ("STRING", {"default": BODY_PART_DEFAULT_W_H[part_name], "multiline": False}) for part_name in s.PART_NAMES}
            }
        }

    RETURN_TYPES = ("TRACKING", "STRING")
    RETURN_NAMES = ("tracking", "prompt")
    FUNCTION = "convert"
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"

    def convert(self, pose_kps, id_include, **parts_width_height):
        parts_width_height = {part_name.replace("_width_height", ''): value for part_name, value in parts_width_height.items()}
        enabled_part_names = [part_name for part_name in self.PART_NAMES if len(parts_width_height[part_name].strip())]
        tracked = {part_name: {} for part_name in enabled_part_names}
        id_include = id_include.strip()
        id_include = list(map(int, id_include.split(','))) if len(id_include) else []
        prompt_string = ''
        pose_kps, max_people = SinglePersonProcess.sort_and_get_max_people(pose_kps)

        for person_idx in range(max_people):
            if len(id_include) and person_idx not in id_include:
                continue
            processor = SinglePersonProcess(pose_kps, person_idx)
            for part_name in enabled_part_names:
                bbox_size = tuple(map(int, parts_width_height[part_name].split(',')))
                part_bboxes = processor.get_xyxy_bboxes(part_name, bbox_size)
                id_coordinates = {idx: part_bbox+(processor.width, processor.height) for idx, part_bbox in part_bboxes.items()}
                tracked[part_name][person_idx] = id_coordinates

        for class_name, class_data in tracked.items():
            for class_id in class_data.keys():
                class_id_str = str(class_id)
                # Use the incoming prompt for each class name and ID
                _class_name = class_name.replace('L', '').replace('R', '').lower()
                prompt_string += f'"{class_id_str}.{class_name}": "({_class_name})",\n'

        return (tracked, prompt_string)

```
