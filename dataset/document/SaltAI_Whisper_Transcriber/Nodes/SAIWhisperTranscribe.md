# Whisper Transcribe
## Documentation
- Class name: `SAIWhisperTranscribe`
- Category: `SALT/Whisper`
- Output node: `False`

This node is designed to transcribe audio and video files using the Whisper model. It processes media files to extract audio, then transcribes the audio into text, providing detailed transcription data including timestamps and potentially images, depending on the media type and transcription settings.
## Input types
### Required
- **`whisper_model`**
    - The Whisper model, processor, and device configuration used for transcription. This affects the accuracy and quality of the transcription.
    - Comfy dtype: `WHISPER_MODEL`
    - Python dtype: `Tuple[torch.nn.Module, Any, torch.device]`
- **`file_path`**
    - The path to the media file to be transcribed. Supports both audio and video files, validating against supported formats.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`frame_rate`**
    - Optional. Specifies the frame rate for audio extraction from video files, affecting the temporal resolution of the transcription.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`chunk_type`**
    - Specifies whether the transcription should be segmented by sentences or words, affecting the granularity of the output timestamps.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`max_new_tokens`**
    - Optional. Limits the number of new tokens generated during transcription, impacting the length and detail of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`transcription_text`**
    - Comfy dtype: `STRING`
    - The complete transcription text without timestamps or segmentation.
    - Python dtype: `str`
- **`transcription_timestamp_dict`**
    - Comfy dtype: `DICT`
    - A structured representation of the transcription with detailed information, including timestamps.
    - Python dtype: `Dict[str, Any]`
- **`transcription_frame_dict`**
    - Comfy dtype: `DICT`
    - Timestamped frames of the transcription, providing temporal context to the text.
    - Python dtype: `Dict[int, str]`
- **`prompt_schedule`**
    - Comfy dtype: `STRING`
    - A schedule of prompts used during transcription, if applicable.
    - Python dtype: `str`
- **`images`**
    - Comfy dtype: `IMAGE`
    - A collection of images extracted from the video, if the media type is video.
    - Python dtype: `List[torch.Tensor]`
- **`transcription_count`**
    - Comfy dtype: `INT`
    - The total number of transcription segments produced.
    - Python dtype: `int`
- **`frame_rate`**
    - Comfy dtype: `INT`
    - The frame rate derived from the audio extraction process.
    - Python dtype: `float`
- **`frame_count`**
    - Comfy dtype: `INT`
    - The total number of frames in the video, if the media type is video.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SAIWhisperTranscribe:
    def __init__(self):
        self.video_extensions = [
            ".3g2", ".3gp", ".3gp2", ".3gpp", ".amv", ".asf", ".avi", ".divx",
            ".drc", ".dv", ".f4v", ".flv", ".m2v", ".m4p", ".m4v", ".mkv",
            ".mov", ".mp4", ".mpe", ".mpeg", ".mpeg2", ".mpeg4", ".mpg",
            ".mpv", ".mxf", ".nsv", ".ogg", ".ogv", ".qt", ".rm", ".rmvb",
            ".roq", ".svi", ".vob", ".webm", ".wmv", ".yuv"
        ]
        self.audio_extensions = [
            ".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a", ".wma"
        ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "whisper_model": ("WHISPER_MODEL",),
                "file_path": ("STRING", {})
            },
            "optional": {
                "frame_rate": ("FLOAT", {"default": 8, "min": 1, "max": 244}),
                "chunk_type": (["sentence", "word"],),
                "max_new_tokens": ("INT", {"min": 1, "max": 4096, "default": 128}),
            },
        }

    RETURN_TYPES = ("STRING", "DICT", "DICT", "STRING", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("transcription_text", "transcription_timestamp_dict", "transcription_frame_dict", "prompt_schedule", "images", "transcription_count", "frame_rate", "frame_count")

    FUNCTION = "transcribe"
    CATEGORY = "SALT/Whisper"

    def transcribe(self, whisper_model, file_path, **kwargs):
        model, processor, device = whisper_model

        media_type = self.validate(file_path)
        if not media_type:
            supported_formats = ', '.join(self.video_extensions + self.audio_extensions)
            raise ValueError(f"Unsupported media file format. Please provide a valid video or audio file: {supported_formats}")

        path = os.path.join(INPUT, file_path)
        audio_path, derived_fps, frame_count, duration = self.extract_audio(path, kwargs.get('frame_rate', 8))

        raw_text, transcription, transcription_frame, prompt_schedule, images = self.transcribe_audio(
            audio_path,
            path,
            derived_fps,
            duration,
            model,
            processor,
            kwargs.get("max_new_tokens", 128),
            media_type,
            kwargs.get("chunk_type", "sentence")
        )

        transcription_count = len(transcription_frame)

        return raw_text, transcription, transcription_frame, prompt_schedule, images, transcription_count, derived_fps, frame_count

    def extract_audio(self, file_path, fps):
        os.makedirs(TEMP, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', dir=TEMP) as tmp_file:
            tmp_file_name = tmp_file.name

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
        return tmp_file_name, fps, frame_count, duration

    def transcribe_audio(self, audio_path, file_path, fps, duration, model, processor, max_new_tokens, media_type="audio", chunk_type="sentence"):
        audio = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)
        samples = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            samples = samples.astype(np.float32) / 2 ** 15
        elif audio.sample_width == 4:
            samples = samples.astype(np.float32) / 2 ** 31

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            return_timestamps=chunk_type,
            max_new_tokens=max_new_tokens,
        )
        result = pipe(samples)

        raw_text = result['text'].strip()
        transcription = {}
        transcription_frame = {}
        images = []
        prompt_schedule = ""

        last_end_time = 0
        segment_offset = 0

        for chunk in result['chunks']:
            text = chunk['text']
            start_time, end_time = chunk['timestamp']

            if start_time < last_end_time:
                segment_offset += last_end_time

            adjusted_start_time = start_time + segment_offset
            frame_number = int(adjusted_start_time * fps)

            transcription[round(adjusted_start_time, ndigits=2)] = text.strip()
            transcription_frame[frame_number] = text.strip()
            prompt_schedule += f'"{frame_number}": "{text.strip()}"' + (",\n" if chunk != result['chunks'][-1] else "\n")

            if media_type == "video":
                img = self.extract_frame(file_path, adjusted_start_time, duration)
                images.append(pil2tensor(img))
            else:
                img = Image.new('RGB', (512, 512), color='black')
                images.append(pil2tensor(img))

            last_end_time = end_time

        images = torch.cat(images, dim=0)

        return raw_text, transcription, transcription_frame, prompt_schedule, images

    def extract_frame(self, file_path, timestamp, video_duration):
        if timestamp > video_duration:
            return Image.new('RGB', (512, 512), color='black')
        with VideoFileClip(file_path) as clip:
            frame = clip.get_frame(timestamp)
        return Image.fromarray(frame)

    def validate(self, file_path):
        if any(file_path.lower().endswith(ext) for ext in self.video_extensions):
            return "video"
        elif any(file_path.lower().endswith(ext) for ext in self.audio_extensions):
            return "audio"
        else:
            return False

```
