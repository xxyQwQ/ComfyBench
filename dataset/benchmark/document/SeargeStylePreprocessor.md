- `SeargeStylePreprocessor`: The SeargeStylePreprocessor node is designed to apply predefined styles to input parameters for further processing. It adjusts the inputs based on the active style name and style definitions, ensuring that the inputs are formatted and ready for the next stages of processing.
    - Inputs:
        - `inputs` (Required): A collection of input parameters that may require styling before further processing. It plays a crucial role in determining how the inputs are modified based on the selected style. Type should be `PARAMETER_INPUTS`.
        - `active_style_name` (Required): Specifies the name of the style to be applied to the inputs. It influences the selection of styling rules that are applied, tailoring the inputs for specific processing needs. Type should be `STRING`.
        - `style_definitions` (Required): Contains the definitions of available styles in a structured format. It is essential for determining how inputs are modified, providing a blueprint for the styling process. Type should be `STRING`.
    - Outputs:
        - `inputs`: The modified collection of input parameters, adjusted according to the selected style. It represents the inputs ready for subsequent processing stages. Type should be `PARAMETER_INPUTS`.
