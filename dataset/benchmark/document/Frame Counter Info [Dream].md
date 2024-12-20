- `Frame Counter Info [Dream]`: Provides detailed information about the current state of a frame counter in an animation context, including completed frames, total frames, and various time-based metrics.
    - Inputs:
        - `frame_counter` (Required): The frame counter object that tracks the current frame, total frames, and timing information for an animation sequence. Type should be `FRAME_COUNTER`.
    - Outputs:
        - `frames_completed`: The number of frames that have been completed. Type should be `INT`.
        - `total_frames`: The total number of frames in the animation sequence. Type should be `INT`.
        - `first_frame`: A boolean indicating if the current frame is the first frame. Type should be `BOOLEAN`.
        - `last_frame`: A boolean indicating if the current frame is the last frame. Type should be `BOOLEAN`.
        - `elapsed_seconds`: The elapsed time in seconds since the animation started. Type should be `FLOAT`.
        - `remaining_seconds`: The estimated remaining time in seconds until the animation completes. Type should be `FLOAT`.
        - `total_seconds`: The total time in seconds for the animation. Type should be `FLOAT`.
        - `completion`: The completion percentage of the animation. Type should be `FLOAT`.
