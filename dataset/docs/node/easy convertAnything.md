- `easy convertAnything`: The `convert` node dynamically converts input data to a specified output type, including string, integer, float, or boolean. This capability enables versatile data type manipulation within workflows, accommodating a wide range of processing requirements.
    - Parameters:
        - `output_type`: Determines the target output data type for the conversion, such as 'string', 'int', 'float', or 'boolean', guiding the transformation of the input data to meet specific needs. Type should be `COMBO[STRING]`.
    - Inputs:
        - `anything`: Represents the input data to be converted, allowing for a broad spectrum of data types to be processed, enhancing the node's adaptability across various inputs. Type should be `*`.
    - Outputs:
        - `*`: The output is the input data converted to the specified output type, enabling further use or processing in the desired format. Type should be `COMBO[STRING]`.