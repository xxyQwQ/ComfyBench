- `Add To Playlist (mtb)`: This node facilitates the addition of videos to a specified playlist, allowing for the dynamic creation and updating of playlists based on user-defined parameters. It supports both relative and absolute path specifications for videos, and can handle persistent storage of playlists across sessions.
    - Inputs:
        - `relative_paths` (Required): Determines whether the paths of videos added to the playlist should be relative to the output directory. This affects how the paths are stored and interpreted, facilitating easier relocation of the playlist and its contents. Type should be `BOOLEAN`.
        - `persistant_playlist` (Required): Indicates whether the playlist should be stored persistently across sessions. A persistent playlist is saved in a common directory, while a non-persistent one is saved in a session-specific directory. Type should be `BOOLEAN`.
        - `playlist_name` (Required): The name of the playlist, which can include formatting options such as an index. This allows for dynamic naming based on the playlist's contents or order. Type should be `STRING`.
        - `index` (Required): An integer used to format the playlist name, enabling the creation of sequentially named playlists or the organization of playlists by index. Type should be `INT`.
    - Outputs: