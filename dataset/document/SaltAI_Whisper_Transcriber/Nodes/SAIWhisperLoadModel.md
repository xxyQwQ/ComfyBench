# Whisper Model Loader
## Documentation
- Class name: `SAIWhisperLoadModel`
- Category: `SALT/Whisper`
- Output node: `False`

The SAIWhisperLoadModel node is designed to load and prepare Whisper models for speech-to-text operations. It supports loading various versions of OpenAI's Whisper models, ensuring they are ready for transcription tasks by initializing the model and its processor according to the specified model version and computational device.
## Input types
### Required
- **`model`**
    - Specifies the version of the Whisper model to be loaded. This parameter determines which pre-trained Whisper model is initialized for speech-to-text processing.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`device`**
    - Defines the computational device ('cuda' or 'cpu') on which the Whisper model will be loaded. This affects the performance and efficiency of the model's operation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `WHISPER_MODEL`
    - Returns a tuple containing the loaded Whisper model, its processor, and the device it's loaded on, ready for speech-to-text transcription.
    - Python dtype: `Tuple[torch.nn.Module, torch.nn.Module, str]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SAIWhisperLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["openai/whisper-large-v2", "openai/whisper-large-v3", "openai/whisper-base", "openai/whisper-large", "openai/whisper-medium", "openai/whisper-small", "openai/whisper-tiny", "distil-whisper/distil-large-v3"],),
            },
            "optional": {
                "device": (["cuda", "cpu"],),
            },
        }

    RETURN_TYPES = ("WHISPER_MODEL",)
    RETURN_NAMES = ("model", "processor")

    FUNCTION = "load_model"
    CATEGORY = "SALT/Whisper"

    def load_model(self, model, device="cuda"):
        """
        match(model):
            case "openai/whisper-large-v2":
                whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(model, cache_dir=WHISPER_MODELS, use_safetensors=True).to(device)
                processor = AutoProcessor.from_pretrained(model)
            case "openai/whisper-large-v3":               
                whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(model, cache_dir=WHISPER_MODELS, use_safetensors=True).to(device)
                processor = AutoProcessor.from_pretrained(model)
            case _:
                raise ValueError(f"The model `{model}` is not supported. Please choose a valid model.")
        """
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(model, cache_dir=WHISPER_MODELS, use_safetensors=True).to(device)
        processor = AutoProcessor.from_pretrained(model)

        return ((whisper_model, processor, device), )

```
