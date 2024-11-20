---
tags:
- Audio
---

# Load Audio
## Documentation
- Class name: `SaltLoadAudio`
- Category: `SALT/AudioViz/Audio`
- Output node: `False`

This node is designed to load audio files from a specified path, making them ready for further processing or analysis within the audio processing pipeline. It serves as the initial step in handling audio data, ensuring that audio files are properly loaded and available for subsequent operations such as editing, analysis, or conversion.
## Input types
### Required
- **`file_path`**
    - Specifies the path to the audio file that needs to be loaded. This is crucial for locating and accessing the audio data for processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`start_seconds`**
    - Defines the starting point of the audio file in seconds from which the audio should be loaded. This allows for partial loading of audio files, facilitating operations that require only a segment of the audio.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`manual_bpm`**
    - Allows for the manual specification of beats per minute (BPM) if known, which can be used for further processing or analysis. This is optional and used primarily when automatic BPM detection is not desired or needs to be overridden.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`frame_rate`**
    - Sets the frame rate for the audio processing, affecting the temporal resolution of the loaded audio. This is crucial for ensuring compatibility with subsequent processing steps that may require a specific frame rate.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`duration_seconds`**
    - Specifies the duration in seconds for which the audio should be loaded from the start_seconds point. This allows for loading only a specific portion of the audio file, useful in scenarios where the entire file is not needed.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The loaded audio data, ready for further processing or analysis. This output is essential for any subsequent audio manipulation or analysis tasks.
    - Python dtype: `bytes`
- **`bpm`**
    - Comfy dtype: `FLOAT`
    - The beats per minute (BPM) of the loaded audio, either detected automatically or set manually. This information is crucial for rhythm-based processing and analysis.
    - Python dtype: `float`
- **`frame_rate`**
    - Comfy dtype: `INT`
    - The frame rate at which the audio was processed. This information is important for ensuring that subsequent processing steps operate on data with consistent temporal resolution.
    - Python dtype: `int`
- **`frame_count`**
    - Comfy dtype: `INT`
    - The total number of frames in the loaded audio segment. This count is essential for operations that involve direct manipulation of audio frames.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltLoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {}),
                "start_seconds": ("FLOAT", {"min": 0.0, "default": 0.0}),
                "manual_bpm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0}),
                "frame_rate": ("INT", {"default": 8, "min": 1, "max": 244}),
            },
            "optional": {
                "duration_seconds": ("FLOAT", {"min": 0.0, "default": 0.0, "optional": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("audio", "bpm", "frame_rate", "frame_count")
    FUNCTION = "load_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio"

    def load_audio(self, file_path, start_seconds, duration_seconds=0.0, manual_bpm=0.0, frame_rate=24.0):
        INPUT = folder_paths.get_input_directory()
        file_path = os.path.join(INPUT, file_path)

        # Load the audio segment (by start/duration)
        audio = self.get_audio(file_path, start_seconds, duration_seconds)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        audio_segment = audio_segment.set_frame_rate(frame_rate)
        duration_ms = int(duration_seconds * 1000) if duration_seconds else len(audio_segment) - int(start_seconds * 1000)

        bpm = self.analyze_bpm(audio, manual_bpm)
        frame_count = int((duration_ms / 1000.0) * frame_rate)

        return (audio, bpm, frame_rate, frame_count)

    def get_audio(self, file, start_time=0, duration=0):
        TEMP = folder_paths.get_temp_directory()
        os.makedirs(TEMP, exist_ok=True)
        
        temp_file_path = None
        try:
            # Create a temporary file in the specified directory
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=TEMP) as temp_file:
                temp_file_path = temp_file.name
            
            args = [ffmpeg_path, "-y", "-v", "error", "-i", file, "-acodec", "pcm_s16le", "-ar", "44100"]
            if start_time > 0:
                args += ["-ss", str(start_time)]
            if duration > 0:
                args += ["-t", str(duration)]
            args += [temp_file_path]

            subprocess.run(args, check=True)

            with open(temp_file_path, "rb") as f:
                audio_data = f.read()

            return audio_data
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def analyze_bpm(self, audio_bytes, manual_bpm=0.0):
        with io.BytesIO(audio_bytes) as audio_file:
            y, sr = librosa.load(audio_file, sr=None)

        if manual_bpm <= 0:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                bpm = float(tempo[0]) if tempo.size > 0 else 0.0
            else:
                bpm = tempo
        else:
            bpm = manual_bpm

        return round(bpm, ndigits=2)

    @staticmethod
    def calculate_file_hash(filename: str):
        try:
            h = hashlib.sha256()
            h.update(filename.encode())
            h.update(str(os.path.getmtime(filename)).encode())
            return h.hexdigest()
        except Exception as e:
            logger.error(e)
            return float("NaN")

    @classmethod
    def IS_CHANGED(cls, file_path, *args, **kwargs):
        INPUT = folder_paths.get_input_directory()
        file_path = os.path.join(INPUT, file_path)
        hash = cls.calculate_file_hash(file_path)
        return hash

```
