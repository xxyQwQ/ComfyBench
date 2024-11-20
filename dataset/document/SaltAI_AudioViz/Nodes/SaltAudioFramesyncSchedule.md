---
tags:
- AnimationScheduling
- Scheduling
---

# Schedule Audio Framesync
## Documentation
- Class name: `SaltAudioFramesyncSchedule`
- Category: `SALT/AudioViz/Audio/Scheduling`
- Output node: `False`

This node is designed to synchronize audio with frame-based visual content, ensuring that audio events align precisely with specific frames in a video or animation sequence. It calculates and schedules audio keyframes based on various parameters such as amplitude control and frame rate, facilitating the creation of dynamic and responsive audiovisual experiences.
## Input types
### Required
- **`audio`**
    - The raw audio data to be processed. It is crucial for determining the synchronization points and dynamics of the audio in relation to the visual content.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`amp_control`**
    - A parameter that influences the amplitude adjustments of the audio, affecting the overall loudness and dynamics of the synchronized output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`amp_offset`**
    - An offset value for the amplitude control, allowing for fine-tuning of the audio's loudness in relation to the visual content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`frame_rate`**
    - The frame rate of the visual content, essential for aligning audio events with the corresponding video frames.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`start_frame`**
    - The starting frame from which the audio synchronization begins, enabling selective synchronization over specific segments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_frame`**
    - The ending frame at which the audio synchronization ends, allowing for precise control over the duration of the audiovisual alignment.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`curves_mode`**
    - Specifies the mode of curve interpolation for easing the transitions between audio keyframes, enhancing the fluidity and naturalness of the audiovisual experience.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`average_sum`**
    - Comfy dtype: `LIST`
    - A list of calculated loudness values for each frame, representing the synchronized audio dynamics over time.
    - Python dtype: `List[float]`
- **`frame_count`**
    - Comfy dtype: `INT`
    - The total number of frames processed for synchronization, indicating the scope of the audiovisual alignment.
    - Python dtype: `int`
- **`frame_rate`**
    - Comfy dtype: `INT`
    - The frame rate used for the synchronization, reaffirming the temporal resolution of the audiovisual content.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioFramesyncSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        easing_fns = list(easing_functions.keys())
        easing_fns.insert(0, "None")
        return {
            "required": {
                "audio": ("AUDIO",),
                "amp_control": ("FLOAT", {"min": 0.1, "max": 1024.0, "default": 1.0, "step": 0.01}),
                "amp_offset": ("FLOAT", {"min": 0.0, "max": 1023.0, "default": 0.0, "step": 0.01}),
                "frame_rate": ("INT", {"min": 1, "max": 244, "default": 8}),
                "start_frame": ("INT", {"min": 0, "default": 0}),
                "end_frame": ("INT", {"min": -1}),
                "curves_mode": (easing_fns,)
            }
        }

    RETURN_TYPES = ("LIST", "INT", "INT")
    RETURN_NAMES = ("average_sum", "frame_count", "frame_rate")

    FUNCTION = "schedule"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Scheduling"

    def dbfs_floor_ceiling(self, audio_segment):
        min_dbfs = 0
        max_dbfs = -float('inf')
        for chunk in audio_segment[::1000]:
            if chunk.dBFS > max_dbfs:
                max_dbfs = chunk.dBFS
            if chunk.dBFS < min_dbfs and chunk.dBFS != -float('inf'):
                min_dbfs = chunk.dBFS
        return min_dbfs, max_dbfs

    def dbfs2loudness(self, dbfs, amp_control, amp_offset, dbfs_min, dbfs_max):
        if dbfs == -float('inf'):
            return amp_offset
        else:
            normalized_loudness = (dbfs - dbfs_min) / (dbfs_max - dbfs_min)
            controlled_loudness = normalized_loudness * amp_control
            adjusted_loudness = controlled_loudness + amp_offset
            return max(amp_offset, min(adjusted_loudness, amp_control + amp_offset))

    def interpolate_easing(self, values, easing_function):
        if len(values) < 3 or easing_function == "None":
            return values
        interpolated_values = [values[0]]
        for i in range(1, len(values) - 1):
            prev_val, curr_val, next_val = values[i - 1], values[i], values[i + 1]
            diff_prev = curr_val - prev_val
            diff_next = next_val - curr_val
            direction = 1 if diff_next > diff_prev else -1
            norm_diff = abs(diff_next) / (abs(diff_prev) + abs(diff_next) if abs(diff_prev) + abs(diff_next) != 0 else 1)
            eased_diff = easing_function(norm_diff) * direction
            interpolated_value = curr_val + eased_diff * (abs(diff_next) / 2)
            interpolated_values.append(interpolated_value)
        interpolated_values.append(values[-1])
        return interpolated_values

    def schedule(self, audio, amp_control, amp_offset, frame_rate, start_frame, end_frame, curves_mode):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        
        frame_duration_ms = int(1000 / frame_rate)
        start_ms = start_frame * frame_duration_ms
        total_length_ms = len(audio_segment)
        total_frames = total_length_ms // frame_duration_ms
        end_ms = total_length_ms if end_frame <= 0 else min(end_frame * frame_duration_ms, total_length_ms)

        audio_segment = audio_segment[start_ms:end_ms]
        dbfs_min, dbfs_max = self.dbfs_floor_ceiling(audio_segment)

        output = {'average': {'sum': []}}

        max_frames = (end_ms - start_ms) // frame_duration_ms
        for frame_start_ms in range(0, (max_frames * frame_duration_ms), frame_duration_ms):
            frame_end_ms = frame_start_ms + frame_duration_ms
            frame_segment = audio_segment[frame_start_ms:frame_end_ms]

            overall_loudness = self.dbfs2loudness(frame_segment.dBFS, amp_control, amp_offset, dbfs_min, dbfs_max)
            output['average']['sum'].append(overall_loudness)

        if curves_mode != "None":
            output['average']['sum'] = self.interpolate_easing(output['average']['sum'], easing_functions[curves_mode])

        output['average']['sum'] = [round(value, 2) for value in output['average']['sum']]

        return (
            output['average']['sum'],
            max_frames,
            frame_rate
        )

```
