# Third-party notices

This file records known upstream components in the Phase 1 release candidate. It is an engineering inventory, not legal advice. Before publication, verify the exact upstream commit, retain required notices, and obtain any additional permission required by the relevant rights holder.

| Component | Upstream | Known terms | Current handling |
|---|---|---|---|
| DiffLocks | https://github.com/Meshcapade/difflocks | Non-commercial scientific research; redistribution and derivative-work restrictions | Not redistributed. Physics setup clones official commit `fcc73747dc60320c30228b6711000a53fc0c9d84`, applies a checksum-verified three-file ControlHair patch, and can invoke the official checkpoint downloader. |
| HairStep | https://github.com/GAP-LAB-CUHK-SZ/HairStep | Creative Commons Attribution-NonCommercial 4.0 | Not redistributed. The preparation script clones official commit `40ef81a213ac32879bc127c1bc1683bbcacc1b09`; checkpoints remain under upstream non-commercial terms. |
| UniAnimate-DiT | https://github.com/ali-vilab/UniAnimate-DiT | No top-level license was identified during the Phase 1 audit | Not redistributed. Setup retrieves commit `61d882c25385042f0cf5bcdaf6853238d9756d68` and applies the checksum-verified ControlHair patch under `third_party/patches/`. |
| DiffSynth-Studio | https://github.com/modelscope/DiffSynth-Studio | Apache-2.0 for the upstream project | Not directly redistributed. It is obtained as part of the pinned UniAnimate-DiT checkout. Audited upstream reference: `fed7b18fac2ed4cb802796eec91970e7659bccde`. |
| DWPose | https://github.com/IDEA-Research/DWPose | Apache-2.0 | Small inference-only source portions remain under `control_signal/`; ONNX weights come from the pinned UniAnimate model snapshot. Audited upstream reference: `3dca5db79d9f9ffdd378753ddf6ec66535aace88`. |
| Blender 4.1.1 | https://download.blender.org/release/Blender4.1/ | GNU GPL | Optional physics setup downloads the official Linux x64 archive and verifies SHA-256 `ab2ea3fe991601a5e6bd2cda786ecaa919c0b39e0550e59978b5d40270c260d3`. Blender is not redistributed. |
| ControlHair Blender templates | `physics_simulation/assets/` | ControlHair project assets containing Blender scene data | Distributed through Git LFS. Setup verifies the original SHA-256 OIDs and byte sizes before installing them into the ignored DiffLocks checkout; embedded third-party components retain their own terms. |
| Wan 2.1 | https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P | Apache-2.0 model card at audited revision | Weights are not redistributed; preparation pins revision `8823af45fcc58a8aa999a54b04be9abc7d2aac98`. |
| UniAnimate-DiT weights | https://huggingface.co/ZheWang123/UniAnimate-DiT | MIT model card at audited revision | Weights are not redistributed; preparation pins revision `77f855e096ca9b1d62387bfd0c3fb2a9691702cf`. |
| ControlHair website examples | https://github.com/linwk20/controlhair-web | Reference portraits are FFHQ-derived; generated controls and outputs are project artifacts | Three triptychs were split into test packages with source commit, derivation, and checksums. |
| FFHQ reference portraits | https://github.com/NVlabs/ffhq-dataset | Dataset materials: CC BY-NC-SA 4.0; individual images retain original Flickr licenses | The three example reference portraits are FFHQ-derived. They are not covered by the project MIT license; individual attribution requirements still apply. |

## Retrieved media

Retrieved third-party repositories may include example portraits, celebrity
images, screenshots, meshes, tokenizer files, or other media with rights that
differ from the surrounding source code. They are ignored by this repository
and remain subject to their upstream terms.

The files under `examples/` identify FFHQ as the reference-portrait source.
Their generated ControlHair controls and outputs are documented separately in
each package metadata file.

All preparation endpoints, revisions, destinations, and observed license labels
are machine-readable in `configs/model_sources.json`. Preparation receipts do
not replace the upstream license texts or grant additional rights.

The Apache-2.0 text retained for audited DWPose and DiffSynth-Studio portions is
available under [`LICENSES/`](LICENSES/README.md).
