# Create Audio Mask
## Documentation
- Class name: `CreateAudioMask`
- Category: `KJNodes/deprecated`
- Output node: `False`

The CreateAudioMask node is designed for generating audio-based visual masks by analyzing the amplitude of audio signals. It utilizes audio data to create dynamic masks that can visually represent the audio's intensity and rhythm, making it suitable for applications requiring audio-visual synchronization or enhancements.
## Input types
### Required
- **`invert`**
    - A boolean flag that, when set to true, inverts the color scheme of the generated masks, offering a visual contrast option.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`frames`**
    - Specifies the number of frames to generate for the mask, affecting the temporal resolution of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`scale`**
    - A scaling factor that adjusts the size of visual elements within the mask, allowing for customization of the mask's visual intensity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`audio_path`**
    - The file path to the audio file from which the mask will be generated, serving as the source for audio analysis.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`width`**
    - Determines the width of each generated mask frame, impacting the spatial resolution of the visual representation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of each generated mask frame, influencing the vertical dimension of the mask's visual output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a series of images representing the audio mask, where each image corresponds to a frame of the mask generated based on the audio input.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CreateAudioMask:
       
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "createaudiomask"
    CATEGORY = "KJNodes/deprecated"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "invert": ("BOOLEAN", {"default": False}),
                 "frames": ("INT", {"default": 16,"min": 1, "max": 255, "step": 1}),
                 "scale": ("FLOAT", {"default": 0.5,"min": 0.0, "max": 2.0, "step": 0.01}),
                 "audio_path": ("STRING", {"default": "audio.wav"}),
                 "width": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
        },
    } 

    def createaudiomask(self, frames, width, height, invert, audio_path, scale):
        try:
            import librosa
        except ImportError:
            raise Exception("Can not import librosa. Install it with 'pip install librosa'")
        batch_size = frames
        out = []
        masks = []
        if audio_path == "audio.wav": #I don't know why relative path won't work otherwise...
            audio_path = os.path.join(script_directory, audio_path)
        audio, sr = librosa.load(audio_path)
        spectrogram = np.abs(librosa.stft(audio))
        
        for i in range(batch_size):
           image = Image.new("RGB", (width, height), "black")
           draw = ImageDraw.Draw(image)
           frame = spectrogram[:, i]
           circle_radius = int(height * np.mean(frame))
           circle_radius *= scale
           circle_center = (width // 2, height // 2)  # Calculate the center of the image

           draw.ellipse([(circle_center[0] - circle_radius, circle_center[1] - circle_radius),
                      (circle_center[0] + circle_radius, circle_center[1] + circle_radius)],
                      fill='white')
             
           image = np.array(image).astype(np.float32) / 255.0
           image = torch.from_numpy(image)[None,]
           mask = image[:, :, :, 0] 
           masks.append(mask)
           out.append(image)

        if invert:
            return (1.0 - torch.cat(out, dim=0),)
        return (torch.cat(out, dim=0),torch.cat(masks, dim=0),)

```
