- `ttN advPlot string`: This node is designed to generate advanced plotting strings based on specified parameters, including node identification, widget interaction, and value formatting. It dynamically constructs plot text elements that can be used to visualize data or control elements in a UI context, emphasizing flexibility in label presentation and numerical value adjustment.
    - Inputs:
        - `node` (Required): Identifies the node to which the plot string will be associated, enabling dynamic interaction with xyPlot options. Type should be `COMBO[STRING]`.
        - `widget` (Required): Specifies the widget to interact with, allowing for dynamic selection of options based on the node's configuration. Type should be `COMBO[STRING]`.
        - `text` (Required): The text input that will be split and formatted according to the delimiter to generate plot values. Type should be `STRING`.
        - `delimiter` (Required): Defines the character or sequence used to split the text input into individual plot values. Type should be `STRING`.
        - `label_type` (Required): Determines the type of label to be used for each plot value, affecting the presentation and information included. Type should be `COMBO[STRING]`.
    - Outputs:
        - `plot_text`: The generated plot text, ready for use in visualization or UI control. Type should be `STRING`.
