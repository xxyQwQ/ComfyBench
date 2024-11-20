---
tags:
- CLIP
- CLIPConditioning
- Conditioning
---

# CLIPTextEncode Scheduled Sequence
## Documentation
- Class name: `SaltCLIPTextEncodeSequence`
- Category: `SALT/AudioViz/Scheduling/Conditioning`
- Output node: `False`

The SaltCLIPTextEncodeSequence node is designed for generating a sequence of conditioning data based on textual descriptions over a specified frame count. It leverages CLIP models to encode text into a format suitable for guiding generative models, with additional parameters to control the normalization and interpretation of token weights. This node is particularly useful for applications requiring dynamic content generation over time, such as animations or video content creation.
## Input types
### Required
- **`clip`**
    - The CLIP model used for text encoding. It plays a crucial role in converting textual descriptions into a format that can be understood by generative models.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`token_normalization`**
    - Specifies the method for normalizing tokens, which can affect the encoding process and the resulting conditioning data. Options include none, mean, length, and length+mean, allowing for flexibility in handling token data.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`weight_interpretation`**
    - Determines how weights are interpreted during the encoding process, with options like comfy, A1111, compel, and comfy++. This parameter influences the generation of conditioning data by adjusting the significance of different tokens.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`frame_count`**
    - The number of frames for which conditioning data is generated. This parameter sets the length of the sequence, enabling the creation of dynamic content over time.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text`**
    - Textual descriptions provided as input, which are encoded into conditioning data. Supports multiline input and dynamic prompts, allowing for detailed and varied content generation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`conditioning_sequence`**
    - Comfy dtype: `CONDITIONING`
    - The generated sequence of conditioning data, suitable for guiding generative models over the specified frame count.
    - Python dtype: `List[torch.Tensor]`
- **`frame_count`**
    - Comfy dtype: `INT`
    - The number of frames for which conditioning data has been generated, confirming the length of the output sequence.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SaltCLIPTextEncodeSequence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++"],),
                "frame_count": ("INT", {"default": 100, "min": 1, "max": 1024, "step": 1}),
                "text": ("STRING", {"multiline": True, "placeholder": '''"0": "A portrait of a rosebud",
"25": "A portrait of a blooming rosebud",
"50": "A portrait of a blooming rose",
"75": "A portrait of a rose"'''}),
            }
        }
        
    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning_sequence", "frame_count")

    FUNCTION = "encode"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Conditioning"

    def encode(self, clip, text, frame_count, token_normalization, weight_interpretation):
        
        try:
            text_dict = json.loads("{"+text+"}")
        except json.JSONDecodeError as e:
            raise ValueError("Unable to decode propmt schedule:", e)

        conditionings = []
        prev_frame_num = 0
        prev_encoded = None
        for frame_num, prompt in sorted(text_dict.items(), key=lambda item: int(item[0])):
            frame_num = int(frame_num)
            if USE_BLK:
                encoded = blk_adv.encode(clip=clip, text=prompt, token_normalization=token_normalization, weight_interpretation=weight_interpretation)
            else:
                encoded = CLIPTextEncode.encode(clip=clip, text=prompt)
            for _ in range(prev_frame_num, frame_num):
                conditionings.append(prev_encoded)
            prev_encoded = [encoded[0][0][0], encoded[0][0][1]]
            prev_frame_num = frame_num
        for _ in range(prev_frame_num, frame_count):
            conditionings.append(prev_encoded)

        conditioning_count = len(conditionings)

        return (conditionings, conditioning_count)

    def cond_easing(self, type, frame_count, conditioning_count):
        if type == "linear":
            return np.linspace(frame_count // conditioning_count, frame_count, conditioning_count, dtype=int).tolist()
        elif type == "sinus":
            t = np.linspace(0, np.pi, conditioning_count)
            sinus_values = np.sin(t)
            normalized_values = (sinus_values - sinus_values.min()) / (sinus_values.max() - sinus_values.min())
            scaled_values = normalized_values * (frame_count - 1) + 1
            unique_keyframes = np.round(scaled_values).astype(int)
            unique_keyframes = np.unique(unique_keyframes, return_index=True)[1]
            return sorted(unique_keyframes.tolist())
        elif type == "sinus_inverted":
            return (np.cos(np.linspace(0, np.pi, conditioning_count)) * (frame_count - 1) + 1).astype(int).tolist()
        elif type == "half_sinus":
            return (np.sin(np.linspace(0, np.pi / 2, conditioning_count)) * (frame_count - 1) + 1).astype(int).tolist()
        elif type == "half_sinus_inverted":
            return (np.cos(np.linspace(0, np.pi / 2, conditioning_count)) * (frame_count - 1) + 1).astype(int).tolist()
        else:
            raise ValueError("Unsupported cond_keyframes_type: " + type)

```
