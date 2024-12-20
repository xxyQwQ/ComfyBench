- `LatentSwitch`: The LatentSwitch node is designed to dynamically select between multiple latent representations based on a given index. It facilitates the manipulation of latent spaces by allowing the selection of specific latent inputs for further processing or output.
    - Inputs:
        - `select` (Required): Specifies the index of the latent representation to be selected. This index determines which latent input ('latent1', etc.) is used for the node's operation. The range of this index starts from 1, allowing for dynamic selection among potentially numerous latent inputs. Type should be `INT`.
        - `sel_mode` (Required): unknown Type should be `BOOLEAN`.
        - `input1` (Optional): The primary latent representation input. This input is crucial for the node's operation as it represents the default or initial latent space to be considered in the absence of additional specified latent inputs. Type should be `*`.
    - Outputs:
        - `selected_value`: The selected latent representation based on the 'select' index. If the index is invalid, 'input1' is returned as the default. Type should be `*`.
        - `selected_label`: The label of the selected latent representation, indicating which latent input was chosen based on the 'select' index. Type should be `STRING`.
        - `selected_index`: The index of the selected latent representation, reflecting the 'select' input value. Type should be `INT`.
