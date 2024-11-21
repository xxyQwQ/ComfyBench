- `VHS_LoadAudioUpload`: The VHS_LoadAudioUpload node is designed for uploading and processing audio files within the Video Helper Suite. It allows users to upload audio files, specifying the desired starting point and duration for processing. This node is essential for integrating audio content into video projects, enabling precise control over the audio segment to be used.
    - Inputs:
        - `audio` (Required): Specifies the audio file to be uploaded from a predefined list of available files. This selection is crucial for determining the specific audio content to be processed. Type should be `COMBO[STRING]`.
        - `start_time` (Required): Determines the starting point, in seconds, from which the audio file should be processed. This parameter allows for selective use of audio content within a larger file. Type should be `FLOAT`.
        - `duration` (Required): Specifies the duration, in seconds, for which the audio from the starting point should be processed. This enables precise control over the segment of the audio file to be used. Type should be `FLOAT`.
    - Outputs:
        - `audio`: Returns the processed audio segment, allowing it to be integrated into video projects or further manipulated. Type should be `VHS_AUDIO`.