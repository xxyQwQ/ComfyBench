- `SaltAudioCompressor`: The SaltAudioCompressor node is designed to compress audio signals, reducing the dynamic range of the audio to ensure a more consistent volume level across the track. It utilizes parameters such as threshold, ratio, attack, and release times to control the compression effect, making it suitable for audio post-production and enhancement tasks.
    - Inputs:
        - `audio` (Required): The raw audio data to be compressed. This input is crucial as it represents the audio signal that will undergo compression, directly influencing the output quality and characteristics. Type should be `AUDIO`.
        - `threshold_dB` (Required): The threshold level in decibels (dB) above which compression is applied. This parameter determines the loudness level at which the compressor starts to reduce the gain, playing a key role in the dynamics control of the audio. Type should be `FLOAT`.
        - `ratio` (Required): The compression ratio, indicating how much the audio signal is reduced once it crosses the threshold. It's essential for defining the intensity of the compression effect. Type should be `FLOAT`.
        - `attack_ms` (Required): The attack time in milliseconds, specifying how quickly the compressor reacts to audio exceeding the threshold. It affects the compressor's responsiveness to sudden loudness changes. Type should be `INT`.
        - `release_ms` (Required): The release time in milliseconds, defining how quickly the compressor stops affecting the audio after it falls below the threshold. This parameter influences the smoothness of the volume level transitions. Type should be `INT`.
    - Outputs:
        - `audio`: The compressed audio data, resulting from the application of dynamic range compression to the input audio. This output showcases the effect of the compression settings on the original audio signal. Type should be `AUDIO`.