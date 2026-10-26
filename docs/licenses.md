# License and access status

Decision D-005 in PLAN.md depends on this file, due at the end of week 2. This is a working
tracker, not legal advice. Fill in the "Verified" column only after reading the actual license
text or terms, and record who read it and when.

| Component | What we know so far | Source of that knowledge | Verified |
| --- | --- | --- | --- |
| SAM 3.1 (checkpoints) | Released 27 March 2026 as an update to SAM 3; checkpoints are on Hugging Face. The SAM 3 license (SAM License, last updated 19 Nov 2025) neither permits nor forbids commercial use, bars military, weapons and ITAR uses, requires passing the license on with redistribution, and requires acknowledging SAM in publications. Not checked whether 3.1 uses the same text. | Summary of the SAM 3 `LICENSE` page on Hugging Face, read 2026-10-26 | Partly: summary only, not the full text |
| SAM 3.1 (Meta Model API) | Model `sam-3.1` on `POST https://api.meta.ai/v1/responses`. Image, video (MP4 or MOV; H.264, H.265, MPEG-4) and text, box or point prompts; returns boxes and masks, with stable track IDs across video frames. $2.50 per 1,000 images; $0.20 per 1,000 video frames, at least 50 frames billed per video request. Rate limit 3,000 requests per minute per team. Video length, frame, resolution and file size limits are not published on the pages read. Data retention and whether inputs are used for training are not stated for this API (the older Llama API says it does not train on customer content, which may not apply). | `dev.meta.ai` docs: models, SAM overview, media segmentation, pricing and rate limits; read 2026-10-26 | Partly: limits and data terms still open |
| SAM 3D Objects | Checkpoints and code are under the SAM License (same family as SAM 3, so the commercial-use and redistribution points above apply, unconfirmed against its own `LICENSE`). Hugging Face access is gated and asks for legal name, date of birth and organization; some requests were rejected. Input is an image plus a mask; output is shape, texture and pose (Gaussian splat PLY in the example). No GPU memory figure published on the model or repository page (see `doc/setup.md`). **Not available on the Meta Model API**: the model list has no 3D model. | Hugging Face model page, GitHub README, `dev.meta.ai/docs/models`; read 2026-10-26 | Partly |
| TRELLIS.2 | README says code and the TRELLIS.2-4B weights are MIT. Rendering depends on NVIDIA `nvdiffrast` and `nvdiffrec`, each under its own license; one news site says those limit use to research or evaluation, not yet checked. Needs an NVIDIA GPU with at least 24 GB, tested on A100 and H100 under Linux only. Not on the Meta Model API. | GitHub README; a Japanese CG news site for the NVIDIA clause; read 2026-10-26 | Partly: NVIDIA licenses not read |
| Vision-language model for material class | Not chosen (D-008). The Meta Model API offers multimodal Muse models that take image and video: `muse-spark-1.3` ($1.25 input, $4.25 output per 1M tokens) and `muse-wonder-1.0` ($4.00 and $20.00). Contributor variants are cheaper but let Meta use content for training, so they must not be used with benchmark videos. | `dev.meta.ai/docs/pricing-rate-limits`, read 2026-10-26 | Partly |
| Reconstruction back end | Not chosen (PLAN.md P1.1). | | No |
| Convex decomposition (CoACD or V-HACD) | Not checked. | | No |
| Isaac Sim and Isaac Lab | Not checked. | | No |

## Questions to answer for each row

* May we use it for the intended purpose (research only, or commercial)?
* May we redistribute outputs, such as meshes generated from it?
* Are there usage limits or costs, and do they fit a 20-scene benchmark plus development?
* Can it be self-hosted, and what GPU memory does it need?

## Findings for P0.7 (2026-10-26)

Read from public pages by Claude Code. Pages were summarised by a fetch tool, so the license
rows are marked partly verified until a human reads the full texts.

* Decisions so far: D-005 is research use. The shape stage plan (D-015) is to run the SAM 3D
  Objects and TRELLIS.2 checkpoints on Modal or Baseten GPUs; neither host has been checked.
* The Meta Model API covers segmentation and tracking (SAM 3.1) and a language model for
  material class. It does not offer SAM 3D Objects or TRELLIS.2. The shape stage therefore
  cannot be API-only. Options for a human to decide: self-host those two on a rented GPU for
  that stage only, use a third-party host (needs its own license and privacy check), or build
  the MVP without the shape stage first (boxes from masks, as the P0.6 baseline does).
* Cost of the API is small: a 60 s clip at 30 fps is 1,800 frames, about $0.36 per pass.
* Open for the Meta Model API: video length, size and resolution limits; data retention;
  whether inputs are used for training in standard mode. Uploading videos (D-004) should wait
  for the terms.
* Open for licenses: full SAM License text for SAM 3.1 and SAM 3D Objects; NVIDIA `nvdiffrast`
  and `nvdiffrec` licenses; SAM 3D GPU memory from `doc/setup.md`; D-005 intended use.
