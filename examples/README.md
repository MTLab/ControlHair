# Examples

Each example package contains:

```text
input.png             # first frame cropped from the left panel
control.mp4           # middle control-signal panel
expected_output.mp4   # right generated-video panel
preview_triptych.mp4  # original composite used for extraction
metadata.json         # provenance, layout, derivation, and checksums
```

The extracted files are intended for integration tests and qualitative inspection. They are decoded or re-encoded derivatives of the website composites, not the original lossless research assets.

## Rebuild an example

Install `ffmpeg` and `ffprobe`, then run:

```bash
python tools/split_triptych.py source.mp4 examples/category/example_id \
  --source-repository https://github.com/linwk20/controlhair-web \
  --source-commit e7416af727367e6bd4e133f25fe6c6c4b29dcfc2 \
  --source-path static/videos/path/to/source.mp4 \
  --source-url https://media.githubusercontent.com/media/linwk20/controlhair-web/e7416af727367e6bd4e133f25fe6c6c4b29dcfc2/static/videos/path/to/source.mp4
```

## Source and license

The reference portraits are sourced from the
[Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset). FFHQ dataset
materials use CC BY-NC-SA 4.0; individual images retain the original Flickr
license and attribution requirements recorded by FFHQ. The package metadata
documents the ControlHair project-page source and all derived artifacts.
