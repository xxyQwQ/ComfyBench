---
tags:
- AnimationScheduling
- Scheduling
---

# Schedule Metrics
## Documentation
- Class name: `SaltKeyframeMetrics`
- Category: `SALT/AudioViz/Scheduling/Util`
- Output node: `False`

The SaltKeyframeMetrics node is designed to compute and visualize key metrics of a keyframe schedule, such as minimum, maximum, sum, average, absolute sum, and absolute average values, along with the duration of the schedule. It aims to provide a comprehensive overview of the schedule's characteristics through numerical analysis and graphical representation, facilitating the evaluation and adjustment of keyframe-based animations or effects.
## Input types
### Required
- **`schedule_list`**
    - The list of keyframe values to be analyzed. It is crucial for determining the metrics and visual representation of the schedule.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
### Optional
- **`start_frame`**
    - The starting frame index for the analysis, allowing for partial evaluation of the schedule.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_frame`**
    - The ending frame index for the analysis. If set to -1 or beyond the schedule list length, it defaults to the length of the schedule list.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_rate`**
    - The frame rate used to calculate the duration of the schedule, enhancing the accuracy of temporal metrics.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`value_min`**
    - Comfy dtype: `FLOAT`
    - The minimum value among the keyframe values within the specified range.
    - Python dtype: `float`
- **`value_max`**
    - Comfy dtype: `FLOAT`
    - The maximum value among the keyframe values within the specified range.
    - Python dtype: `float`
- **`value_sum`**
    - Comfy dtype: `FLOAT`
    - The sum of all keyframe values within the specified range.
    - Python dtype: `float`
- **`value_avg`**
    - Comfy dtype: `FLOAT`
    - The average of the keyframe values within the specified range.
    - Python dtype: `float`
- **`abs_sum`**
    - Comfy dtype: `FLOAT`
    - The sum of the absolute values of the keyframe values within the specified range, providing insight into the overall activity level regardless of direction.
    - Python dtype: `float`
- **`abs_avg`**
    - Comfy dtype: `FLOAT`
    - The average of the absolute values of the keyframe values within the specified range, offering a measure of the general intensity.
    - Python dtype: `float`
- **`duration`**
    - Comfy dtype: `FLOAT`
    - The total duration of the specified range of keyframes, calculated based on the frame rate.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltKeyframeMetrics:
    @classmethod
    def INPUT_TYPES(cls):
        input_types = {
            "required": {
                "schedule_list": ("LIST",),
            },
            "optional": {
                "start_frame": ("INT", {"min": 0, "default": 0}),
                "end_frame": ("INT", {"min": 0, "default": -1}),
                "frame_rate": ("FLOAT", {"min": 0.1, "default": 24.0}),
            }
        }
        return input_types

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("value_min", "value_max", "value_sum", "value_avg", "abs_sum", "abs_avg", "duration")

    FUNCTION = "schedule_metrics"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Util"

    def schedule_metrics(self, schedule_list, start_frame=0, end_frame=-1, frame_rate=24.0):
        if end_frame == -1 or end_frame > len(schedule_list):
            end_frame = len(schedule_list)

        num_frames = max(2, end_frame - start_frame)
        keyframe_values = schedule_list[start_frame:end_frame]

        value_min = float(np.min(keyframe_values).round(2))
        value_max = float(np.max(keyframe_values).round(2))
        value_sum = float(np.sum(keyframe_values).round(2))
        value_avg = float(np.mean(keyframe_values).round(2))
        abs_sum = float(np.sum(np.abs(keyframe_values)).round(2))
        abs_avg = float(np.mean(np.abs(keyframe_values)).round(2))
        duration = num_frames / frame_rate

        return value_min, value_max, value_sum, value_avg, abs_sum, abs_avg, duration
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

```
