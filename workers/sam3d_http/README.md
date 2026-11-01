# SAM 3D over HTTP

A reference server for `shape.transport: http`. Any server with the same wire format works; the
format is in `src/video2sim/models/sam3d_http.py`. Not yet run on a GPU (see the notes in
`server.py` and the Dockerfile).

1. Get the gated checkpoint `facebook/sam-3d-objects` from Hugging Face into a directory
   `weights/checkpoints/` that holds `pipeline.yaml`. Record the revision.
2. Build and run:

```bash
docker build -t video2sim-sam3d workers/sam3d_http
```

```bash
docker run --gpus all -p 8000:8000 -v "$PWD/weights:/weights" -e SAM3D_REVISION=<revision> video2sim-sam3d
```

3. Point the project at it in a copy of the config (`--config my.yaml`):

```yaml
shape:
  transport: http
  http:
    url: http://localhost:8000
    api_key_env: null   # set SAM3D_API_KEY on the server and name the client's variable here
```

4. Check it with `uv run video2sim doctor --config my.yaml`.

The server serves one repo id (`shape.sam3d.checkpoint` must equal `SAM3D_REPO_ID`) and one draw
at a time. Run several replicas behind a load balancer and raise `shape.parallel_samples` to use
them.
