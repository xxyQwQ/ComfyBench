# Whisper Transcribe (OpenAI API)
## Documentation
- Class name: `SAIOpenAIAPIWhisper`
- Category: `SALT/Whisper`
- Output node: `False`

The SAIOpenAIAPIWhisper node is designed to transcribe audio files using OpenAI's Whisper model through an API. It abstracts the complexity of audio processing and transcription, providing an easy-to-use interface for converting speech in audio files into text.
## Input types
### Required
- **`file_path`**
    - The path to the audio file to be transcribed. This is crucial for locating and processing the audio file for transcription.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`openai_key`**
    - The API key for accessing OpenAI's Whisper model. This key is essential for authenticating and authorizing the transcription request.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`model`**
    - Specifies the Whisper model version to use for transcription. This allows for flexibility in choosing the model that best fits the transcription needs.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`mode`**
    - Defines the operation mode, such as transcribing. It determines how the audio file will be processed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`language`**
    - The language of the audio to be transcribed. This helps in optimizing the transcription accuracy by informing the model of the language context.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`response_format`**
    - The format in which the transcription results are returned, e.g., text. It specifies the desired output format for the transcription.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`temperature`**
    - Controls the creativity of the transcription. A higher temperature can result in more varied transcriptions.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`timestamp_granularities`**
    - Specifies the granularity of timestamps in the transcription, such as by segment. This affects how detailed the timestamp information in the transcription will be.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`max_frames`**
    - The maximum number of frames to process, which can limit the scope of transcription for longer audio files.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seek_seconds`**
    - The number of seconds to skip from the beginning of the audio file. This allows for starting the transcription at a specific point in the audio.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`prompt`**
    - An optional prompt to guide the transcription process. This can influence the context or focus of the transcription.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`transcription_result`**
    - Comfy dtype: `STRING`
    - The transcribed text of the audio file, providing a textual representation of the spoken content.
    - Python dtype: `str`
- **`audio_path`**
    - Comfy dtype: `STRING`
    - The path to the processed audio file, useful for further analysis or processing.
    - Python dtype: `str`
- **`frames_count`**
    - Comfy dtype: `INT`
    - The total number of frames processed during the transcription, indicating the scope of the transcription.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SAIOpenAIAPIWhisper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {}),
                "openai_key": ("STRING", {}),
            },
            "optional": {
                "model": (["whisper-1"],),
                "mode": (["transcribe", "translate_to_english"],),
                "language": ("STRING", {}),
                "response_format": (["text", "json", "verbose_json", "prompt_schedule"],),
                "temperature": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 0.7}),
                "timestamp_granularities": (["segment", "word"],),
                "max_frames": ("INT", {"default": 0, "min": 0}),
                "seek_seconds": ("FLOAT", {"default": 0.0}),
                "prompt": ("STRING", {"multiline": True, "placeholder": "Optional prompt..."})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("transcription_result", "audio_path", "frames_count")

    FUNCTION = "whisper_v2"
    CATEGORY = "SALT/Whisper"

    def whisper_v2(self, file_path, openai_key, model="whisper-1", mode="transcribe", language="", response_format="text", temperature=0.7, timestamp_granularities="segment", max_frames=0, seek_seconds=0.0, prompt=""):
        language = language if language else None
        prompt = prompt if prompt else None
        file_path = os.path.join(INPUT, file_path)

        if not os.path.exists(file_path):
            raise ValueError(f"The specified file `{file_path}` does not exist!")
        
        if model not in ("whisper-1"):
            raise ValueError(f"The specified model `{model}` does not exist!")

        if mode not in ("transcribe", "translate_to_english"):
            logger.error(f'The `mode` selected "{mode}" is not valid. Please use either "transcribe", or "translate_to_english"')
            mode = "transcribe"

        openai.api_key = openai_key

        audio_path, fps, total_frames = self.extract_audio(file_path)

        max_frames = max_frames if max_frames != 0 else total_frames

        match mode:
            case "transcribe":
                if response_format == "prompt_schedule":
                    transcription = self.transcribe_audio(file_path, model, prompt, language, "verbose_json", temperature, timestamp_granularities, json_string=False)
                    out = self.prompt_schedule(transcription, fps, max_frames, seek_seconds)
                else:
                    out = self.transcribe_audio(file_path, model, prompt, language, response_format, temperature, timestamp_granularities)
            case "translate_to_english":
                out = self.translate_audio(file_path, model, prompt, response_format, temperature)

        return (out, audio_path, total_frames)

    def transcribe_audio(self, file_path, model="whisper-1", prompt=None, language=None, response_format="json", temperature=0.7, timestamp_granularities="segment", json_string=True):
        with open(file_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                model=model,
                file=audio_file,
                prompt=prompt,
                response_format=response_format,
                language=language,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities
            )
            if response_format in ("json", "verbose_json"):
                segments = getattr(response, 'segments', [])
                if json_string:
                    out = json.dumps(segments, ensure_ascii=True, indent=4)
                else:
                    out = segments
            else:
                out = response
            
            return out

    def translate_audio(self, file_path, model="whisper-1", prompt=None, response_format="json", temperature=0.7):
        with open(file_path, "rb") as audio_file:
            response = openai.audio.translations.create(
                model=model,
                file=audio_file,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
            )
            if response_format in ("json", "verbose_json"):
                segments = getattr(response, 'segments', [])
                out = json.dumps(segments, ensure_ascii=True, indent=4)
            else:
                out = response
            
            return out

    def extract_audio(self, file_path):
        os.makedirs(TEMP, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', dir=TEMP) as tmp_file:
            tmp_file_name = tmp_file.name

        try:
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                file_path
            ]
            result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            fps_str = result.stdout.decode().strip()
            if fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den
            else:
                fps = 21
        except Exception as e:
            fps = 21

        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-i', file_path,
            '-vn',  # No video
            '-acodec', 'mp3',
            '-ar', '16000',  # Sample rate for whisper
            '-ac', '1',  # Mono channel for whisper 
            tmp_file_name
        ]
        subprocess.run(cmd, check=True)
        audio = AudioSegment.from_file(tmp_file_name)
        duration = audio.duration_seconds
        frame_count = int(duration * fps)
        return tmp_file_name, fps, frame_count
    
    def prompt_schedule(self, transcription, fps, max_frames, seek_seconds):
        prompt_schedule = ""
        max_seconds = max_frames / fps if max_frames > 0 else float('inf')
        start_frame = int(seek_seconds * fps)

        if isinstance(transcription, list):
            for idx, segment in enumerate(transcription):
                segment_start = segment.get("start", 0.0)
                if segment_start < seek_seconds or segment_start > max_seconds:
                    continue

                frame_number = int(segment_start * fps) - start_frame
                text = segment.get("text", "")
                if frame_number >= 0:
                    prompt_schedule += f'"{frame_number}": "{text.strip()}"' + (",\n" if idx != len(transcription) else "\n")

        return prompt_schedule

```
