- `ttN pipe2BASIC`: The ttN pipe2BASIC node is designed to simplify the structure of a given pipeline by extracting and repackaging its core components into a basic pipeline format. This process facilitates easier manipulation and understanding of the pipeline's fundamental elements.
    - Parameters:
    - Inputs:
        - `pipe`: The 'pipe' input is the pipeline to be simplified, containing various components such as models, clips, and VAEs. It serves as the primary data structure for transformation into a basic pipeline format. Type should be `PIPE_LINE`.
    - Outputs:
        - `basic_pipe`: The 'basic_pipe' output is a simplified version of the input pipeline, containing only its essential components such as model, clip, VAE, and positive and negative conditioning. Type should be `BASIC_PIPE`.
        - `pipe`: The 'pipe' output returns the original pipeline as received in the input, allowing for further manipulation or inspection. Type should be `PIPE_LINE`.