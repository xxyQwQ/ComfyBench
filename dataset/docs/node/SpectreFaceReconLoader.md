- `SpectreFaceReconLoader`: The SpectreFaceReconLoader node is designed to initialize and load the Spectre model along with a face tracker for facial recognition tasks. It prepares the model for subsequent operations by downloading necessary models and setting configuration parameters, ensuring the system is ready for face tracking and recognition within images or video streams.
    - Parameters:
        - `fp16`: Determines whether the model should be loaded in half-precision (FP16) format, which can reduce memory usage and potentially increase performance on compatible hardware. Type should be `BOOLEAN`.
    - Inputs:
    - Outputs:
        - `spectre_model`: Returns a tuple containing the initialized face tracker and Spectre model, ready for facial recognition tasks. Type should be `SPECTRE_MODEL`.