# Control-signal extraction

`python -m control_signal.run <simulation_dir>` converts the optional Blender
simulation outputs into the strand-orientation and DWPose conditioning video
used by ControlHair diffusion.

The extraction uses the rendered hair mask, so training-data face detection,
YOLOFace cropping, and generic human-parsing annotation are not part of this
release. HairStep and DWPose remain runtime dependencies for users generating
new control sequences. See the optional physics section in
[`README.md`](../README.md).
