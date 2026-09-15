import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "sample_sync.py"
SPEC = importlib.util.spec_from_file_location("sample_sync", MODULE_PATH)
sample_sync = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sample_sync
SPEC.loader.exec_module(sample_sync)


class SampleSyncTest(unittest.TestCase):
    def config(self, root):
        config = dict(sample_sync.DEFAULTS)
        config.update(
            runs_root=str(root / "runs"),
            state_db=str(root / "state.sqlite3"),
            stable_seconds=0,
        )
        return config

    def make_run(self, root):
        samples = root / "runs" / "experiment_a" / "run_1" / "samples"
        samples.mkdir(parents=True)
        (samples / "_eval_meta.json").write_text(json.dumps({
            "samples": {"case_a": {
                "task_type": "v2v", "edit_type": "add", "prompt": "add item",
                "dataset": "source_a", "index": 7,
            }}
        }))
        for name in (
            "step_001000_case_a.mp4",
            "step_001000_case_a_src_in.mp4",
            "step_001000_case_a_tgt_in.mp4",
            "_ref0_case_a.mp4",
        ):
            (samples / name).write_bytes(name.encode())
        return samples

    def test_parse_step_roles(self):
        self.assertEqual(sample_sync.parse_step_file(Path("step_001000_case_a.mp4")), (1000, "case_a", "result_video"))
        self.assertEqual(sample_sync.parse_step_file(Path("step_001000_case_a_src_in.mp4")), (1000, "case_a", "source_video"))
        self.assertEqual(sample_sync.parse_step_file(Path("step_001000_case_a_tgt_in.mp4")), (1000, "case_a", "target_video"))

    def test_discover_groups_case_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root)
            rows = list(sample_sync.discover_rows(self.config(root)))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row.sample_id, "experiment_a:run_1:001000:case_a")
            self.assertEqual(row.metadata["prompt"], "add item")
            self.assertEqual(set(row.media), {"result_video", "source_video", "target_video", "reference_video"})
            self.assertEqual(row.media["result_video"].cos_key, "training-eval/experiment_a/run_1/step_001000_case_a.mp4")

    def test_media_stability_and_change_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = self.make_run(root)
            config = self.config(root)
            media = next(sample_sync.discover_rows(config)).media["result_video"]
            state = sample_sync.StateStore(root / "state.sqlite3")
            try:
                self.assertFalse(state.media_ready(media, 100, 60))
                self.assertTrue(state.media_ready(media, 160, 60))
                state.mark_media_success(media, "cos://bucket/key")
                self.assertEqual(state.media_uri(media), "cos://bucket/key")
                path = samples / "step_001000_case_a.mp4"
                path.write_bytes(b"changed")
                changed = next(sample_sync.discover_rows(config)).media["result_video"]
                self.assertFalse(state.media_ready(changed, 200, 60))
            finally:
                state.close()

    def test_row_payload_contains_links_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root)
            row = next(sample_sync.discover_rows(self.config(root)))
            state = sample_sync.StateStore(root / "state.sqlite3")
            try:
                for role, media in row.media.items():
                    state.media_ready(media, 0, 0)
                    state.mark_media_success(media, f"cos://bucket/{role}")
                payload = sample_sync.row_payload(row, state)
                self.assertEqual(payload["train_step"], 1000)
                self.assertEqual(payload["source_index"], 7)
                self.assertEqual(payload["result_video"], "cos://bucket/result_video")
            finally:
                state.close()

    @patch.object(sample_sync.requests, "post")
    @patch.object(sample_sync.requests, "get")
    def test_upload_uses_tokenless_cos_api(self, get, post):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.mp4"
            path.write_bytes(b"video")
            config = self.config(Path(tmp))
            media = sample_sync.media_for(path, config, "exp", "run")
            get.return_value = Mock(status_code=404)
            response = Mock(status_code=201)
            response.json.return_value = {
                "cos_uri": sample_sync.cos_uri(config, media), "size": media.size
            }
            response.raise_for_status.return_value = None
            post.return_value = response
            uri = sample_sync.upload_media(config, media)
            self.assertEqual(uri, sample_sync.cos_uri(config, media))
            args, kwargs = post.call_args
            self.assertTrue(args[0].endswith("/api/cos/upload"))
            self.assertNotIn("Authorization", kwargs["headers"])


if __name__ == "__main__":
    unittest.main()
