- `SaltStringPathDirname`: This node extracts the directory path from a given file path, effectively isolating the portion of the path that specifies the directory containing the file. It serves to simplify file path manipulation by providing a straightforward method to obtain the directory component.
    - Inputs:
        - `file_path` (Required): The file path from which the directory name is to be extracted. This input is crucial for determining the specific directory location within a filesystem. Type should be `STRING`.
    - Outputs:
        - `path_directory`: The directory portion of the provided file path, which specifies the location of the file within the filesystem. Type should be `STRING`.
