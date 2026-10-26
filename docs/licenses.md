# License and access status

Decision D-005 in PLAN.md depends on this file, due at the end of week 2. This is a working
tracker, not legal advice. Fill in the "Verified" column only after reading the actual license
text or terms, and record who read it and when.

| Component | What we know so far | Source of that knowledge | Verified |
| --- | --- | --- | --- |
| SAM 3.1 (checkpoints) | Not needed under D-004 (hosted by Meta). The `facebookresearch/sam3` `LICENSE` is the SAM License, last updated 19 Nov 2025. It never names a version, and the Hugging Face SAM 3.1 card says only "License: other", so that it covers 3.1 is likely but not stated. Grant (1(a)): non-exclusive, worldwide, non-transferable, royalty-free, limited. No commercial or research-only wording. 1(b)(i): pass the license on when distributing. 1(b)(ii): acknowledge SAM in published research. 1(b)(iv): no reverse engineering. 1(b)(v): no military, warfare or trade-control end uses. | Raw `LICENSE` file, read through a fetch tool with quotes, 2026-10-26 | Yes for the text; not checked by a lawyer |
| SAM 3.1 (Meta Model API) | Official terms, acceptable use policy and data-handling page were not found on `dev.meta.ai` (`/terms` returns 404, the docs index links none). Third-party trackers of the older Llama API terms say content sent through unpaid services may be used to train Meta's models, that Meta may retain content its systems flag as a policy violation, and that outputs may not be used to train a competing model. Treat these as unverified, and use a paid account only. Model `sam-3.1` on `POST https://api.meta.ai/v1/responses`. Image, video (MP4 or MOV; H.264, H.265, MPEG-4) and text, box or point prompts; returns boxes and masks, with stable track IDs across video frames. $2.50 per 1,000 images; $0.20 per 1,000 video frames, at least 50 frames billed per video request. Rate limit 3,000 requests per minute per team. Video length, frame, resolution and file size limits are not published on the pages read. Data retention and whether inputs are used for training are not stated for this API (the older Llama API says it does not train on customer content, which may not apply). | `dev.meta.ai` docs: models, SAM overview, media segmentation, pricing and rate limits; read 2026-10-26 | No: terms not found; limits and data terms still open |
| SAM 3D Objects | Same SAM License text, 19 Nov 2025; its patent grant refers to "the SAM 3 Materials". No research-only or commercial wording, no clause on cloud or third-party hosting; 1(b)(i) applies only if we distribute the materials. Running the checkpoint on Modal or Baseten for our own jobs is not addressed either way, so treat it as allowed but unconfirmed. Outputs are "as is", with the risk on us. Hugging Face access is gated (legal name, date of birth, organization) and some requests were rejected. Access granted to the project owner on 2026-10-26 (their report). No GPU memory figure on the model or repository page (see `doc/setup.md`). **Not on the Meta Model API.** | Raw `LICENSE` file; Hugging Face and GitHub pages; 2026-10-26 | Yes for the text; hosting question open |
| TRELLIS.2 | README: code and TRELLIS.2-4B weights are MIT. Its rendering needs `nvdiffrast` (NVIDIA Source Code License, 1-Way Commercial) and `nvdiffrec` (NVIDIA Source Code License). Both say in 3.3: "only may be used or intended for use non-commercially", where non-commercial means "research or evaluation purposes only". So the MIT label does not make a pipeline using TRELLIS.2 commercial-safe. Needs an NVIDIA GPU with at least 24 GB, tested on A100 and H100 under Linux. Not on the Meta Model API. | GitHub README; raw NVIDIA `LICENSE` files; 2026-10-26 | Yes for the text; not checked by a lawyer |
| Modal (hosted GPUs for D-015) | SaaS Agreement, effective May 2026. Customer owns Customer Data (§1.8). Modal may not train AI models on it without written consent (§5.5.4). Data processed by the service is expunged promptly after processing (§5.4). Nothing on third-party model licenses or gated weights. Use is for the customer's internal business only (§1.1), so the team must not give outside people direct access to the service. Prices per second: A10 $0.000306, L40S $0.000542, A100 80 GB $0.000694, H100 $0.001097. Starter plan has $30 a month free credit; academics can apply for up to $10k. SOC 2 on all plans. | `modal.com/legal/terms` and `modal.com/pricing`, read 2026-10-26 | Yes for the text |
| Baseten (hosted GPUs for D-015) | Terms last updated 15 Sep 2026. Customer owns Customer Content (§6.1). Baseten will not use Customer Content, models, outputs or inference logs to train or fine-tune (§6.3). May use usage telemetry to improve the service (§6.4). Content is not logged only on services designated Zero Data Retention (§2.5); otherwise deletion is on written request (§7.4). Third-party and open-source models stay subject to their own licenses (§1), so the SAM License and the NVIDIA terms still bind us. | `baseten.co/terms-and-conditions`, read 2026-10-26 | Yes for the text |
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

## D-005 outcome (2026-10-26)

Intended use is research. Under that:

* The SAM License allows it, subject to acknowledging SAM in publications and the end-use bans.
* `nvdiffrast` and `nvdiffrec` allow research or evaluation only, so TRELLIS.2 is fine for this
  project but blocks any later commercial use. A commercial version would need a replacement
  renderer or a licence from NVIDIA, and a fresh check of the SAM License.
* Papers or reports from this project must acknowledge SAM.
* Modal and Baseten terms (read 2026-10-26): neither trains on our content, neither forbids
  running these checkpoints, and Baseten says model licenses still bind us. Neither has been
  tried with SAM 3D or TRELLIS.2, so D-015 stays open until a checkpoint runs on one of them.
* Still open: the official Meta Model API terms (training use, retention, outputs), and credits
  for Modal or Baseten (see D-015). Gated Hugging Face access to SAM 3D is granted.
