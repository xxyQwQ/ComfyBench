- `Time Seed (JPS)`: The Time Seed node generates a unique seed value based on the current time, optionally allowing for a fixed seed to be specified. This functionality is crucial for ensuring reproducibility in processes that require randomization, by providing a way to initialize random number generators with a consistent starting point.
    - Inputs:
        - `fixed_seed` (Required): Specifies a fixed seed value to be used instead of generating one based on the current time. If set to 0, a time-based seed is generated, ensuring variability with each execution. Type should be `INT`.
    - Outputs:
        - `seed`: The generated seed value, which is either based on the current time or the specified fixed seed, used for initializing random number generators. Type should be `INT`.
