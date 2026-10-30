"""Adapters around external models (SAM 3.1, SAM 3D, a vision-language model).

One module per model, each exposing a small typed interface. Rules:

* Import heavy libraries lazily, inside the function that needs them, so that this package
  imports on a machine without a GPU.
* Record the model name and checkpoint id in the run manifest.
* Take a seed for anything that samples, and record it.
* No model weights in git.
"""
