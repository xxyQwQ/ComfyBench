- `SaveAudio`: The SaveAudio node is designed for saving audio data to disk in FLAC format. It allows for the inclusion of metadata, such as prompts and additional information, within the saved audio files. This node is particularly useful for persisting audio outputs with contextual metadata, facilitating easier organization and retrieval of generated audio content.
    - Inputs:
        - `audio` (Required): The audio data to be saved. This includes the waveform and sample rate of the audio to be persisted. Type should be `AUDIO`.
        - `filename_prefix` (Required): An optional prefix for the filename under which the audio will be saved, allowing for custom naming conventions. Type should be `STRING`.
    - Outputs: