"""Judge gateway: lets a judge run the pipeline with a personal token instead of model keys.

Three routes mirror the three model endpoints of ``configs/default.yaml``, each forwarded to a
fixed upstream: VLM chat completions to Baseten, SAM 3.1 video tracking to the Meta Model API,
SAM 3D draws to the Modal worker. The real keys exist only in the host's secret store. See
``workers/gateway/README.md`` for the runbook and ``src/video2sim/gateway_profile.py`` for the
client side (``--gateway``).
"""
