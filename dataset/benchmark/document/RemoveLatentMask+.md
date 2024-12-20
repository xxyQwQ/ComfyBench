- `RemoveLatentMask+`: This node is designed to remove the noise mask from a given set of latent samples. It ensures that the latent samples are cleaned of any previously applied noise masks, maintaining the integrity of the original latent representation.
    - Inputs:
        - `samples` (Required): The latent samples from which the noise mask is to be removed. This operation is crucial for processes that require the original, unaltered state of the latent samples. Type should be `LATENT`.
    - Outputs:
        - `latent`: The cleaned latent samples, with the noise mask removed, ready for further processing or generation tasks. Type should be `LATENT`.
