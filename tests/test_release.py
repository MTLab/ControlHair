from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_model_manifest(self) -> None:
        manifest = json.loads(
            (ROOT / "configs" / "model_sources.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], 1)
        profiles = manifest["environment"]["profiles"]
        self.assertEqual(profiles["paper"]["pytorch"], "2.5.0")
        self.assertEqual(profiles["blackwell"]["pytorch"], "2.12.1")
        self.assertEqual(profiles["blackwell"]["cuda"], "13.0")
        self.assertEqual(
            manifest["models"]["controlhair"]["revision"],
            "e37f6d39397fa24fbe694a075f22e29968a62b36",
        )
        controlhair = manifest["models"]["controlhair"]
        self.assertEqual(controlhair["inference_tensors"], 826)
        self.assertEqual(controlhair["inference_tensor_bytes"], 2455193360)
        self.assertEqual(
            controlhair["download_patterns"],
            ["model-controlhair-inference.safetensors"],
        )
        self.assertEqual(
            manifest["datasets"]["controlhair_480p"]["revision"],
            "1d97c73ff695c75545a93c6714006c726d4961e0",
        )
        unianimate = manifest["source_repositories"]["unianimate_dit"]
        self.assertEqual(
            unianimate["destination"], "third_party/UniAnimate-DiT"
        )
        self.assertEqual(
            unianimate["patch"],
            "third_party/patches/unianimate-61d882c-controlhair.patch",
        )
        difflocks = manifest["source_repositories"]["difflocks"]
        self.assertEqual(
            difflocks["patch"],
            "third_party/patches/difflocks-fcc737-controlhair.patch",
        )
        self.assertNotIn("yolo_face", manifest["models"])
        self.assertNotIn("hairstep_sam", manifest["models"])
        for source in manifest["source_repositories"].values():
            revision = source["revision"]
            self.assertRegex(revision, r"^[0-9a-f]{40}$")
        for name in ("wan", "unianimate"):
            self.assertRegex(manifest["models"][name]["revision"], r"^[0-9a-f]{40}$")

    def test_third_party_sources_are_not_vendored(self) -> None:
        forbidden = [
            "uni_hair_train_and_inference",
            "uni_hair_data_preparation",
            "third_party/UniAnimate-DiT",
            "third_party/difflocks",
            "third_party/HairStep",
        ]
        if (ROOT / ".git").exists():
            result = subprocess.run(
                ["git", "ls-files", "-z", "--", *forbidden],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
            self.assertEqual(result.stdout, b"", "Restricted source is tracked by Git")
        else:
            for relative in forbidden:
                self.assertFalse((ROOT / relative).exists(), f"Restricted source is vendored: {relative}")

    def test_experimental_windmap_pipeline_is_removed(self) -> None:
        if (ROOT / ".git").exists():
            result = subprocess.run(
                ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
            )
            named_paths = [
                item.decode() for item in result.stdout.split(b"\0")
                if item and "windmap" in Path(item.decode()).name.lower()
            ]
        else:
            named_paths = [path for path in ROOT.rglob("*") if "windmap" in path.name.lower()]
        self.assertEqual(named_paths, [])
        patch = ROOT / "third_party" / "patches" / "unianimate-61d882c-controlhair.patch"
        self.assertNotIn("windmap", patch.read_text(encoding="utf-8").lower())

    def test_unianimate_retrieve_and_patch_files(self) -> None:
        patch_dir = ROOT / "third_party" / "patches"
        patch = patch_dir / "unianimate-61d882c-controlhair.patch"
        manifest = patch_dir / "unianimate-61d882c-controlhair.sha256"
        setup = (ROOT / "scripts" / "setup_unianimate.sh").read_text(encoding="utf-8")
        self.assertTrue(patch.is_file())
        self.assertTrue(manifest.is_file())
        self.assertIn("61d882c25385042f0cf5bcdaf6853238d9756d68", setup)
        self.assertIn("git clone --filter=blob:none --no-checkout", setup)
        self.assertIn("git -C \"${upstream_dir}\" apply", setup)
        self.assertEqual(len(manifest.read_text(encoding="utf-8").splitlines()), 4)

    def test_difflocks_retrieve_and_patch_files(self) -> None:
        patch_dir = ROOT / "third_party" / "patches"
        patch = patch_dir / "difflocks-fcc737-controlhair.patch"
        manifest = patch_dir / "difflocks-fcc737-controlhair.sha256"
        setup = (ROOT / "scripts" / "setup_difflocks.sh").read_text(encoding="utf-8")
        self.assertTrue(patch.is_file())
        self.assertTrue(manifest.is_file())
        self.assertIn("fcc73747dc60320c30228b6711000a53fc0c9d84", setup)
        self.assertIn("git -C \"${upstream_dir}\" apply", setup)
        self.assertEqual(len(manifest.read_text(encoding="utf-8").splitlines()), 3)

    def test_optional_physics_assets(self) -> None:
        setup = (ROOT / "scripts" / "setup_physics.sh").read_text(encoding="utf-8")
        assets = (ROOT / "physics_simulation" / "prepare_assets.py").read_text(encoding="utf-8")
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("blender-4.1.1-linux-x64.tar.xz", setup)
        self.assertIn('lfs pull --include="physics_simulation/assets/*.blend"', setup)
        self.assertIn("--download-checkpoints", setup)
        self.assertIn("third_party/difflocks/download_checkpoints.sh", setup)
        self.assertIn("ab2ea3fe991601a5e6bd2cda786ecaa919c0b39e0550e59978b5d40270c260d3", setup)
        self.assertIn("089f0dff771d26e4b0e70bad50b7207617946f83af6f2680ac059986e2f0cfde", assets)
        self.assertIn("089cc16a3fd6cc00700328e06bf49d223ca7d8a9ffd592a0e67907aca6b7da6b", assets)
        self.assertIn("physics_simulation/assets/*.blend filter=lfs", attributes)

    def test_project_and_example_licenses(self) -> None:
        self.assertTrue((ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License"))
        for metadata_path in ROOT.glob("examples/*/*/metadata.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            license_info = metadata["license"]
            self.assertEqual(license_info["status"], "ffhq-source-confirmed")
            self.assertEqual(license_info["dataset_license"], "CC-BY-NC-SA-4.0")

    def test_release_document_local_links(self) -> None:
        documents = list(ROOT.glob("*.md"))
        documents += [
            ROOT / "examples" / "README.md",
            ROOT / "inference" / "README.md",
            ROOT / "control_signal" / "README.md",
        ]
        for document in documents:
            text = document.read_text(encoding="utf-8")
            links = re.findall(r"\[[^]]+\]\(([^)#]+)", text)
            links += re.findall(r'(?:href|src)="([^"]+)"', text)
            for link in links:
                if link.startswith(("http://", "https://", "mailto:")):
                    continue
                target = document.parent / link
                self.assertTrue(target.exists(), f"Broken local link in {document}: {link}")

    def test_no_known_sensitive_patterns(self) -> None:
        patterns = re.compile(
            r"(sig=|[?&]sv=|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|"
            r"pixalgorithm\.blob\.core\.windows\.net|Weikai-Internship25|weikai-Vol)"
        )
        text_suffixes = {
            ".bib", ".cfg", ".css", ".html", ".ini", ".json", ".md",
            ".py", ".sh", ".toml", ".txt", ".yaml", ".yml",
        }
        ignored_roots = {
            ".venv", ".venv-release", ".venv-physics", "artifacts",
            "datasets", "models", "third_party",
        }
        if (ROOT / ".git").exists():
            result = subprocess.run(
                ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
            )
            paths = [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]
        else:
            paths = [
                path for path in ROOT.rglob("*")
                if path.is_file()
                and not ignored_roots.intersection(path.relative_to(ROOT).parts)
            ]
        for path in paths:
            if not path.is_file() or path.suffix not in text_suffixes:
                continue
            if path.resolve() == Path(__file__).resolve():
                # This test contains the detection expressions themselves.
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.assertIsNone(patterns.search(text), f"Sensitive pattern in {path}")

    def test_no_large_tracked_files(self) -> None:
        if (ROOT / ".git").exists():
            result = subprocess.run(
                ["git", "ls-files", "-s", "-z"], cwd=ROOT, check=True, capture_output=True
            )
            for item in result.stdout.split(b"\0"):
                if not item:
                    continue
                metadata, path = item.decode().split("\t", 1)
                object_id = metadata.split()[1]
                size = int(
                    subprocess.run(
                        ["git", "cat-file", "-s", object_id],
                        cwd=ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout
                )
                self.assertLess(size, 20 * 1024 * 1024, path)
        else:
            ignored_roots = {
                ".venv", ".venv-release", "artifacts", "datasets", "models", "third_party"
            }
            paths = [
                path
                for path in ROOT.rglob("*")
                if path.is_file() and not ignored_roots.intersection(path.relative_to(ROOT).parts)
            ]
            for path in paths:
                self.assertLess(path.stat().st_size, 20 * 1024 * 1024, str(path))

    def test_preparation_dry_runs(self) -> None:
        subprocess.run(
            [
                sys.executable,
                "scripts/prepare_models.py",
                "--component", "wan",
                "--dry-run",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/prepare_models.py",
                "--component", "controlhair",
                "--accept-third-party-licenses",
                "--dry-run",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/prepare_dataset.py",
                "--component", "controlhair_480p",
                "--accept-private-dataset-terms",
                "--dry-run",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/prepare_external_dependencies.py",
                "--component", "all",
                "--accept-restricted-licenses",
                "--dry-run",
            ],
            cwd=ROOT,
            check=True,
        )

    def test_setup_dry_run(self) -> None:
        for profile in ("paper", "blackwell"):
            subprocess.run(
                ["bash", "scripts/setup.sh", "--profile", profile, "--dry-run"],
                cwd=ROOT,
                check=True,
            )
        subprocess.run(
            [
                "bash", "scripts/setup_physics.sh", "--profile", "paper",
                "--accept-restricted-licenses", "--skip-blender", "--dry-run",
            ],
            cwd=ROOT,
            check=True,
        )

    def test_inference_wrapper_requires_checkpoint(self) -> None:
        result = subprocess.run(
            ["bash", "inference/run_example.sh"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("CONTROLHAIR_CHECKPOINT", result.stderr)

    def test_short_inference_controls_are_exposed(self) -> None:
        patch = (
            ROOT / "third_party" / "patches" / "unianimate-61d882c-controlhair.patch"
        ).read_text(encoding="utf-8")
        self.assertIn("--num_frames", patch)
        self.assertIn("--num_inference_steps", patch)
        self.assertIn("--num_persistent_param_in_dit", patch)
        self.assertIn("num_frames=max_frames", patch)

    def test_controlhair_10k_mp4_training_loader(self) -> None:
        patch = (
            ROOT / "third_party" / "patches" / "unianimate-61d882c-controlhair.patch"
        ).read_text(encoding="utf-8")
        self.assertIn("def load_training_frames", patch)
        self.assertIn('path_dir, "frame.pkl", "clip_001.mp4"', patch)
        self.assertIn('path_dir, "cond.pkl", "condition.mp4"', patch)
        self.assertIn("No valid training samples found", patch)
        self.assertIn('"--max_steps"', patch)
        launcher = (ROOT / "training" / "train.sh").read_text(encoding="utf-8")
        self.assertIn('--max_steps "${MAX_STEPS:-10000}"', launcher)

    def test_sharded_training_checkpoint_loader(self) -> None:
        patch = (
            ROOT / "third_party" / "patches" / "unianimate-61d882c-controlhair.patch"
        ).read_text(encoding="utf-8")
        self.assertIn('key.startswith("pipe.")', patch)
        self.assertIn('key.startswith("pipe.dit.") and "lora" in key', patch)
        self.assertIn("Loaded {len(self.state_dict_new)} LoRA tensors", patch)


if __name__ == "__main__":
    unittest.main()
