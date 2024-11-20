---
tags:
- AnimationScheduling
- Scheduling
---

# Schedule Visualizer
## Documentation
- Class name: `SaltKeyframeVisualizer`
- Category: `SALT/AudioViz/Scheduling/Util`
- Output node: `True`

The SaltKeyframeVisualizer node is designed to visualize the keyframe scheduling process, providing a graphical representation of keyframe values across frames. It calculates various metrics for the keyframes within a specified frame range and generates a visualization image that highlights these metrics, aiding in the analysis and adjustment of keyframe-based animations.
## Input types
### Required
- **`schedule_list`**
    - A list of keyframe values to be visualized. It is crucial for understanding the distribution and variation of keyframe values across the specified frame range, impacting the visualization's accuracy and detail.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
### Optional
- **`start_frame`**
    - The starting frame number for the visualization. It defines the initial point of the frame range to be visualized, affecting the scope of the keyframe analysis.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_frame`**
    - The ending frame number for the visualization. It marks the boundary of the frame range to be analyzed and visualized, determining the extent of the keyframe metrics calculation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`simulate_stereo`**
    - Indicates whether to simulate stereo visualization for the keyframes, adding a dimensional aspect to the analysis by mirroring keyframe values.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`frame_rate`**
    - The frame rate used to calculate the duration of the keyframe sequence. It influences the time-based metrics and the overall temporal context of the visualization.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`schedule_list_a`**
    - An alternative list of keyframe values, providing additional context or variations for visualization.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`schedule_list_b`**
    - Another variation of keyframe values list, offering further insights or comparative analysis in the visualization.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`schedule_list_c`**
    - A third variation of keyframe values list, allowing for a broader range of comparative visualization and analysis.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
## Output types
- **`ui`**
    - A UI component that displays the generated keyframe visualization image. It provides a visual summary of the keyframe metrics and their distribution across the specified frame range, facilitating easier interpretation and analysis.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltKeyframeVisualizer:
    @classmethod
    def INPUT_TYPES(cls):
        input_types = {
            "required": {
                "schedule_list": ("LIST",),
            },
            "optional": {
                "start_frame": ("INT", {"min": 0, "default": 0}),
                "end_frame": ("INT", {"min": 0, "default": -1}),
                "simulate_stereo": ("BOOLEAN", {"default": False}),
                "frame_rate": ("INT", {"min": 1, "default": 24}),
                "schedule_list_a": ("LIST", {"default": None}),
                "schedule_list_b": ("LIST", {"default": None}),
                "schedule_list_c": ("LIST", {"default": None}),
            }
        }
        return input_types

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "visualize_keyframes"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Util"

    def visualize_keyframes(self, schedule_list, start_frame=0, end_frame=-1, simulate_stereo=False, frame_rate=24.0, schedule_list_a=None, schedule_list_b=None, schedule_list_c=None):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)

        schedule_lists = [schedule_list, schedule_list_a, schedule_list_b, schedule_list_c]
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        schedule_names = ['Schedule List', 'Schedule List A', 'Schedule List B', 'Schedule List C']
        metrics_data = []

        plt.figure(figsize=(10, 14 if simulate_stereo else 8))

        for i, sched_list in enumerate(schedule_lists):
            if sched_list is not None:
                if end_frame == -1 or end_frame > len(sched_list):
                    end_frame = len(sched_list)

                num_frames = max(2, end_frame - start_frame)
                frames = np.linspace(start_frame, end_frame - 1, num=num_frames, endpoint=True)
                keyframe_values = np.array(sched_list[start_frame:end_frame])

                plt.plot(frames, keyframe_values, color=colors[i], linewidth=0.5, label=schedule_names[i] + ' Left')
                if simulate_stereo:
                    plt.plot(frames, -keyframe_values, color=colors[i], linewidth=0.5, linestyle='dashed', label=schedule_names[i] + ' Right')
                    plt.fill_between(frames, keyframe_values, 0, color=colors[i], alpha=0.3)
                    plt.fill_between(frames, -keyframe_values, 0, color=colors[i], alpha=0.3)

                metrics = {
                    "Max": np.round(np.max(keyframe_values), 2),
                    "Min": np.round(np.min(keyframe_values), 2),
                    "Sum": np.round(np.sum(keyframe_values), 2),
                    "Avg": np.round(np.mean(keyframe_values), 2),
                    "Abs Sum": np.round(np.sum(np.abs(keyframe_values)), 2),
                    "Abs Avg": np.round(np.mean(np.abs(keyframe_values)), 2),
                    "Duration": (num_frames / frame_rate)
                }
                metrics_data.append((schedule_names[i], metrics))

        plt.title('Schedule Visualization')
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.legend(loc='upper right')

        metrics_text = "Metric Values:\n"
        for name, data in metrics_data:
            metrics_text += f"{name}: "
            metrics_text += ' | '.join([f"{k}: {v}" for k, v in data.items()])
            metrics_text += "\n"

        plt.figtext(0.5, -0.2 if simulate_stereo else -0.1, metrics_text, ha="center", fontsize=12,
                    bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5}, wrap=True)

        filename = str(uuid.uuid4()) + "_keyframe_visualization.png"
        file_path = os.path.join(TEMP, filename)

        plt.savefig(file_path, bbox_inches="tight", pad_inches=1 if simulate_stereo else 0.1)
        plt.close()

        return {
            "ui": {
                "images": [
                    {
                        "filename": filename,
                        "subfolder": "",
                        "type": "temp"
                    }
                ]
            }
        }
    
    @staticmethod
    def gen_hash(input_dict):
        sorted_json = json.dumps(input_dict, sort_keys=True)
        hash_obj = hashlib.sha256()
        hash_obj.update(sorted_json.encode('utf-8'))
        return hash_obj.hexdigest()
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return cls.gen_hash(kwargs)

```
