"""The six pipeline stages, in order. Each module documents its contract and is implemented in
the phase named in its docstring (see PLAN.md). A stage reads typed input plus files from the
run folder and writes typed output plus files. It does not read anything else.

    capture -> reconstruction -> segmentation -> shape -> physics -> export
"""
