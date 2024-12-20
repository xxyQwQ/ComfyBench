- `VidDirIterator`: The VidDirIterator node is designed to navigate through directories containing video files, allowing for the retrieval of video file paths based on their index. This functionality facilitates the organization and selection of video content within a specified directory, streamlining the process of accessing and utilizing video files in various applications.
    - Inputs:
        - `directory_path` (Required): Specifies the path to the directory containing video files. This path is crucial for the node to locate and list the video files for further operations. Type should be `STRING`.
        - `video_index` (Required): Determines the index of the video file to retrieve from the sorted list of video files in the directory. This index is used to select a specific video file, enabling targeted access to video content. Type should be `INT`.
    - Outputs:
        - `string`: Returns the path to the video file at the specified index within the directory. This output facilitates direct access to the selected video file for further processing or playback. Type should be `STRING`.
