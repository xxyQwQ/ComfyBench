- `ChangeLatentBatchSize __Inspire`: This node is designed to modify the batch size of a given latent representation. It achieves this by resizing the tensor associated with the latent samples according to the specified new batch size and mode, ensuring the latent's structure is maintained while adapting to the new batch size requirements.
    - Parameters:
        - `batch_size`: Specifies the target batch size for the latent representation, directly influencing the resizing operation. Type should be `INT`.
        - `mode`: Determines the method of resizing, offering flexibility in how the batch size adjustment is performed. Type should be `COMBO[STRING]`.
    - Inputs:
        - `latent`: The latent representation to be resized. It is crucial for maintaining the integrity of the data while adjusting its batch size. Type should be `LATENT`.
    - Outputs:
        - `latent`: The resized latent representation, now conforming to the specified batch size. Type should be `LATENT`.