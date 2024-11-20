# NormalizedAmplitudeToFloatList
## Documentation
- Class name: `NormalizedAmplitudeToFloatList`
- Category: `KJNodes/audio`
- Output node: `False`

This node is designed to interface with the AudioScheduler nodes, converting normalized amplitude values into a list of floats. It ensures that the amplitude values are within a specific range before conversion, facilitating their use in audio processing workflows.
## Input types
### Required
- **`normalized_amp`**
    - Represents the normalized amplitude values to be converted. It is crucial for ensuring the amplitude values are within the acceptable range [0, 1] before conversion, affecting the node's execution and the accuracy of the output.
    - Comfy dtype: `NORMALIZED_AMPLITUDE`
    - Python dtype: `numpy.ndarray`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The output is a list of floats derived from the normalized amplitude values, suitable for further processing in audio-related applications.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class NormalizedAmplitudeToFloatList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "normalized_amp": ("NORMALIZED_AMPLITUDE",),
                     },}

    CATEGORY = "KJNodes/audio"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "convert"
    DESCRIPTION = """
Works as a bridge to the AudioScheduler -nodes:  
https://github.com/a1lazydog/ComfyUI-AudioScheduler  
Creates a list of floats from the normalized amplitude.
"""

    def convert(self, normalized_amp):
        # Ensure normalized_amp is an array and within the range [0, 1]
        normalized_amp = np.clip(normalized_amp, 0.0, 1.0)
        return (normalized_amp.tolist(),)

```
