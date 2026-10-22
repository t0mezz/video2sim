# License and access status

Decision D-005 in PLAN.md depends on this file, due at the end of week 2. This is a working
tracker, not legal advice. Fill in the "Verified" column only after reading the actual license
text or terms, and record who read it and when.

| Component | What we know so far | Source of that knowledge | Verified |
| --- | --- | --- | --- |
| SAM 3.1 (checkpoints) | Released 27 March 2026 as an update to SAM 3; checkpoints are on Hugging Face. License terms not yet read. | Press coverage and the Hugging Face model card | No |
| SAM 3.1 (Meta Model API) | Available through the API with bearer-token authentication. Limits, pricing and terms not yet read. | Press coverage | No |
| SAM 3D Objects | One source says checkpoints are released for research use. Not confirmed against the license text. | Third-party mirror page | No |
| TRELLIS.2 | MIT license, according to its repository and model pages. | GitHub repository | No |
| Vision-language model for material class | Not chosen. The original brief names the Meta Model API. | Project brief | No |
| Reconstruction back end | Not chosen (PLAN.md P1.1). | | No |
| Convex decomposition (CoACD or V-HACD) | Not checked. | | No |
| Isaac Sim and Isaac Lab | Not checked. | | No |

## Questions to answer for each row

* May we use it for the intended purpose (research only, or commercial)?
* May we redistribute outputs, such as meshes generated from it?
* Are there usage limits or costs, and do they fit a 20-scene benchmark plus development?
* Can it be self-hosted, and what GPU memory does it need?
