"""HTTP contract tests. The model transport is mocked; no GPU is needed."""
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fastapi.testclient import TestClient
from playground import create_app


class PlaygroundTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())
        self.body = {"state": "The answer is A.", "question": "Which answer?", "options": ["A", "B"]}

    def test_home_and_five_examples(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        examples = self.client.get("/api/examples").json()
        self.assertEqual(len(examples), 5)
        self.assertEqual(len({e["id"] for e in examples}), 5)
        for e in examples:
            self.assertLess(e["expected_index"], len(e["options"]))

    def test_invalid_input_is_rejected_before_encoding(self):
        with patch("playground.encode_request") as encode:
            for changes in ({"options": ["A"]}, {"options": ["A", " A "]}, {"state": "  "},
                            {"options": ["A", 7]}, {"options": ["A", "x" * 1001]},
                            {"question": "x" * 4001}, {"unknown": True}):
                with self.subTest(changes=changes):
                    self.assertEqual(self.client.post("/api/decide", json=self.body | changes).status_code, 422)
            encode.assert_not_called()

    def test_cross_origin_and_oversized_body(self):
        self.assertEqual(self.client.post("/api/decide", json=self.body, headers={"origin": "https://example.com"}).status_code, 403)
        self.assertEqual(self.client.post("/api/decide", content=b"x" * 65537).status_code, 413)

    def test_success_returns_measured_result(self):
        response = {"data": [{"data": [[0.9], [0.1]]}]}
        with patch("playground.encode_request", return_value={"input": [[1, 2, 3]]}), patch("playground.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            r = self.client.post("/api/decide", json=self.body)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["selected"], "A")
        self.assertEqual(data["tokens"], 3)
        self.assertGreaterEqual(data["elapsed_ms"], data["inference_ms"])

    def test_invalid_model_output_is_not_presented(self):
        for rows in ([[float("nan")], [0.1]], [[0.8], [0.8]], [[1.0]], [[0.9, 0], [0.1, 0]]):
            with patch("playground.encode_request", return_value={"input": [[1]]}), patch("playground.urllib.request.urlopen", return_value=io.BytesIO(json.dumps({"data": [{"data": rows}]}).encode())):
                self.assertEqual(self.client.post("/api/decide", json=self.body).status_code, 502)

    def test_unavailable_model(self):
        with patch("playground.urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            self.assertFalse(self.client.get("/api/health").json()["ready"])
            with patch("playground.encode_request", return_value={"input": [[1]]}):
                self.assertEqual(self.client.post("/api/decide", json=self.body).status_code, 503)

    def test_encoder_limits_are_returned_to_client(self):
        with patch("playground.encode_request", side_effect=ValueError("Input exceeds token limit")):
            self.assertEqual(self.client.post("/api/decide", json=self.body).status_code, 422)

    def test_backend_is_local_only(self):
        for url in ("https://example.com", "http://example.com", "http://127.0.0.1:18089/path"):
            with self.assertRaises(ValueError):
                create_app(url)


if __name__ == "__main__":
    unittest.main()
