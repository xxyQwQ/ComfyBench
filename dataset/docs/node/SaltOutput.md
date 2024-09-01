- `SaltOutput`: The SaltOutput node is designed to handle various types of output data, including images, audio, and text, formatting and saving them appropriately based on the specified output type. It supports a wide range of file formats and is capable of generating complex UI structures to represent the output data effectively.
    - Parameters:
        - `output_name`: Specifies the name of the output, which is used as the base for generating filenames and identifiers in the saved output files. Type should be `STRING`.
        - `output_desc`: Provides a description for the output, which may be used for metadata or UI display purposes. Type should be `STRING`.
        - `output_type`: Determines the format of the output file (e.g., PNG, JPEG, MP3, WAV, STRING), influencing how the output data is processed and saved. Type should be `COMBO[STRING]`.
        - `animation_fps`: Specifies the frames per second for animated outputs, allowing control over the animation speed. Type should be `INT`.
        - `animation_quality`: Defines the quality level of the animation (DEFAULT or HIGH), affecting the output file's visual fidelity. Type should be `COMBO[STRING]`.
    - Inputs:
        - `output_data`: The actual data to be output, which can vary widely in type (e.g., bytes for audio, torch.Tensor for images, string for text outputs). Type should be `*`.
        - `video_audio`: Optional audio data for video outputs, specifying the audio track to be included in video files. Type should be `AUDIO`.
    - Outputs: