# Inference

The public inference wrapper accepts a prepared input/control example and runs
the 480p ControlHair video diffusion stage. It intentionally does not run the
separately licensed DiffLocks and Blender simulation stage.

Prepare Wan 2.1 and an authorized ControlHair checkpoint as described in the
main [`README.md`](../README.md), then run:

```bash
export CONTROLHAIR_CHECKPOINT=/path/to/composed/controlhair/weights
export WAN_MODEL_DIR="$PWD/models/wan/Wan2.1-I2V-14B-720P"
bash inference/run_example.sh
```

The default example is `examples/motion_control/motion_01`. Pass a different
example and output directory as positional arguments:

```bash
bash inference/run_example.sh \
  examples/wind_control/wind_01 \
  artifacts/inference/wind_01
```

The wrapper creates the legacy UniAnimate condition-frame layout and writes the
generated triptych to `wan_480P_trial_0.mp4`. The published paper reports 853
seconds for the video diffusion stage on an NVIDIA RTX Pro 6000 Max-Q GPU; this
is an offline, high-memory workflow rather than a real-time demo.
