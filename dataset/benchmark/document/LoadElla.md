- `LoadElla`: The LoadElla node is designed to initialize and load the ELLA model along with a specified T5 model into the system, setting them up for further text encoding and processing tasks. It encapsulates the functionality to load model state dictionaries and configure the models for operation on the designated device and data type.
    - Inputs:
        - `ella_model` (Required): Specifies the directory path to the ELLA model to be loaded. This path is crucial for locating and initializing the ELLA model for subsequent operations. Type should be `COMBO[STRING]`.
        - `t5_model` (Required): Indicates the directory path to the T5 model to be loaded alongside ELLA. The T5 model is essential for text embedding processes that precede ELLA's conditioning tasks. Type should be `COMBO[STRING]`.
    - Outputs:
        - `ella`: Returns a dictionary containing the initialized ELLA and T5 models, ready for text encoding and processing tasks. Type should be `ELLA`.