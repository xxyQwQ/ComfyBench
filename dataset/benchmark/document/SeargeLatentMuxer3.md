- `SeargeLatentMuxer3`: The SeargeLatentMuxer3 node is designed for selecting one of three provided latent inputs based on a specified selector input. It facilitates dynamic control over which latent input to proceed with in a computational graph, allowing for flexible manipulation of latent representations in generative models.
    - Inputs:
        - `input0` (Required): The first latent input option for selection. Type should be `LATENT`.
        - `input1` (Required): The second latent input option for selection. Type should be `LATENT`.
        - `input2` (Required): The third latent input option for selection. Type should be `LATENT`.
        - `input_selector` (Required): An integer selector that determines which of the three latent inputs to use. The selector ranges from 0 to 2, with each number corresponding to one of the latent inputs. Type should be `INT`.
    - Outputs:
        - `output`: The selected latent input, based on the value of the input selector. Type should be `LATENT`.