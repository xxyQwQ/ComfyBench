---
tags:
- Scheduling
---

# Schedule Value Threshold
## Documentation
- Class name: `SaltThresholdSchedule`
- Category: `SALT/AudioViz/Scheduling`
- Output node: `False`

This node applies a threshold to a schedule list, modifying each value based on a specified fade type and strength, starting from an optional index. It's designed to create dynamic adjustments in scheduling sequences, allowing for the implementation of fade-in, fade-out, or both effects to enhance the temporal dynamics of a schedule.
## Input types
### Required
- **`float_schedule`**
    - The list of schedule values to be modified. It serves as the base for applying the threshold adjustments, determining the sequence of values that will undergo fading effects.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`max_frames`**
    - Specifies the maximum number of frames for the schedule, defining the temporal limit for the application of the threshold.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`output_mode`**
    - Determines the format of the output schedule list, allowing for different representations such as 'prompt_schedule', 'float_list', 'int_list', or 'raw'.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`schedule_values`**
    - A string representing scheduled values in a JSON format, which are used to generate the sequence with applied thresholds.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`value_schedule_list`**
    - Comfy dtype: `*`
    - The modified schedule list after applying the specified threshold, showcasing the dynamic adjustments made to the original sequence.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltThresholdSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float_schedule": ("LIST", ),
                "max_frames": ("INT", {"min": 1, "max": 4096, "default": 16}),
                "output_mode": (["prompt_schedule", "float_list", "int_list", "raw"],),
                "schedule_values": ("STRING", {"multiline": True, "default": '''"0.0": "A beautiful forest, (green:1.2) color scheme",
"0.5": "A beautiful forest, (autumn:1.2) color scheme",
"1.0": "A beautiful forest, (winter:1.2) color scheme"'''}),
            }
        }

    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("value_schedule_list",)

    FUNCTION = "generate_sequence"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def generate_sequence(self, float_schedule, schedule_values, max_frames, output_mode):
        try:
            text_dict = json.loads("{" + schedule_values + "}")
        except json.JSONDecodeError as e:
            raise ValueError("Unable to decode prompt schedule:", e)

        prompt_sequence = []
        prompt_dict = dict(sorted({float(key): value for key, value in text_dict.items()}.items(), key=lambda x: x[0]))
        default_prompt = next(iter(prompt_dict.values())) if prompt_dict else None

        adjusted_float_schedule = [float_schedule[i] if i < len(float_schedule) else 0 for i in range(max_frames)]

        for float_val in adjusted_float_schedule:
            closest_prompt = default_prompt
            min_diff = float('inf')
            
            for prompt_key, prompt_val in prompt_dict.items():
                diff = abs(float_val - prompt_key)
                
                if diff < min_diff:
                    closest_prompt = prompt_val
                    min_diff = diff

            prompt_sequence.append(closest_prompt)

        if output_mode == "prompt_schedule":
            output = ", ".join(f'"{i}": "{prompt}"' for i, prompt in enumerate(prompt_sequence))
        elif output_mode == "float_list":
            output = [float(value) for value in prompt_sequence]
        else:
            output = [int(value) for value in prompt_sequence]

        return (output, )  

```
