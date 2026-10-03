"""Сквозной тест check.py на фейковом API (сеть и деньги не нужны): распознавание → условие → проверка →
отчёт → признаки. Проверяет кэш, защиту настроек, постраничное сохранение при сбое и --with-key.
Запуск из папки проекта:  python3 -m unittest discover tests -v"""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent


class FakeAPI(BaseHTTPRequestHandler):
    calls = []          # (тип запроса, reasoning, есть ли эталонные ответы в тексте)
    fail_next_ocr = 0   # сколько следующих запросов распознавания отвергнуть
    texts = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["messages"][0]["content"][0]["text"]
        kind = "ocr" if "решений по математике" in text[:120] else "task" if text.startswith("Перед тобой") else "check"
        if kind == "ocr" and "сканер" in text[:60]:
            kind = "ocr2"
        FakeAPI.calls.append((kind, (body.get("reasoning") or {}).get("effort"), "ЭТАЛОННЫЕ ОТВЕТЫ" in text))
        FakeAPI.texts.append(text)
        if kind == "ocr" and FakeAPI.fail_next_ocr:
            FakeAPI.fail_next_ocr -= 1
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "image rejected"}')
            return
        if kind == "ocr2":  # второе прочтение (промпт v2) видит 2+2=6 — расхождение
            content = {"lines": ["N1: 2+2=6", "N2: x=3"], "drawings": [], "task_labels": ["1", "2"], "legibility": "good"}
        elif kind == "ocr":
            content = {"lines": ["N1: 2+2=5", "N2: x=3"], "drawings": [], "task_labels": ["1", "2"], "legibility": "good"}
        elif kind == "task":
            content = {"tasks": [{"task_no": "1", "text": "2+2", "figure": ""}, {"task_no": "2", "text": "x-3=0", "figure": ""}]}
        else:
            content = {"results": [
                {"task_no": "1", "verdict": "incorrect", "student_answer": "5", "correct_answer": "4",
                 "errors": [{"type": "computational", "explanation": "2+2=4"}]},
                {"task_no": "2", "verdict": "correct", "student_answer": "3", "correct_answer": "3", "errors": []}]}
        data = json.dumps({"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)},
                                        "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 1000, "completion_tokens": 200,
                                     "completion_tokens_details": {"reasoning_tokens": 0}}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPI)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.tmp = Path(tempfile.mkdtemp())
        root = cls.tmp
        for f in ("check.py", "features.py", "ocr_eval.py"):
            shutil.copy(PROJECT / f, root / f)
        (root / ".env").write_text(f"LLM_BASE_URL=http://127.0.0.1:{cls.server.server_port}/v1\nLLM_API_KEY=test\n")
        d = root / "data"
        (d / "anonymized" / "W901").mkdir(parents=True)
        for name in ("solution_1.jpg", "solution_2.jpg", "task_1.jpg"):
            Image.new("RGB", (60, 80), "white").save(d / "anonymized" / "W901" / name)
        write_csv(d / "works.csv", ["work_id", "student_id", "redacted"], [["W901", "S901", "1"]])
        write_csv(d / "files.csv", ["work_id", "file", "role"],
                  [["W901", "solution_1.jpg", "solution"], ["W901", "solution_2.jpg", "solution"],
                   ["W901", "task_1.jpg", "task"]])
        write_csv(d / "labels.csv", ["work_id", "task_no", "is_correct", "error_step", "error_type", "is_consequence",
                                     "explanation", "notes"],
                  [["W901", "1", "0", "", "", "", "2+2=5", ""], ["W901", "2", "1", "", "", "", "", ""]])
        write_csv(d / "splits_new.csv", ["folder", "split", "task_key", "source"], [["f901", "dev", "T901", ""]])
        write_csv(d / "private" / "work_map.csv", ["folder", "work_id"], [["f901", "W901"]])
        write_csv(d / "keys" / "answer_key.csv", ["task_key", "task_no", "works", "answer", "answer_type", "method",
                                                  "confidence", "comment"],
                  [["T901", "1", "W901", "4", "number", "sympy", "high", ""],
                   ["T901", "2", "W901", "3", "number", "sympy", "high", ""]])
        (d / "models.json").write_text(json.dumps([{"id": "test/model", "input_modalities": ["text", "image"],
                                                    "supported_parameters": ["reasoning", "response_format"],
                                                    "pricing": {"input_rub_per_million": 100,
                                                                "output_rub_per_million": 500}}]))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_cli(self, *args):
        r = subprocess.run([sys.executable, "check.py", "--model", "test/model", *args], cwd=self.tmp,
                           capture_output=True, text=True, timeout=120)
        return r.stdout + r.stderr

    def test_1_page_failure_keeps_paid_pages(self):
        FakeAPI.fail_next_ocr = 1
        out = self.run_cli("--works", "W901", "--steps", "transcribe", "--max-rub", "5")
        self.assertIn("не распознаны страницы", out)
        pages = list((self.tmp / "runs" / "test_model" / "W901" / "pages").glob("*.json"))
        self.assertEqual(len(pages), 1)  # одна страница оплачена и сохранена
        FakeAPI.calls.clear()
        self.run_cli("--works", "W901", "--steps", "transcribe", "--max-rub", "5")
        self.assertEqual([c[0] for c in FakeAPI.calls], ["ocr"])  # докачана только упавшая

    def test_2_full_run_and_cache(self):
        out = self.run_cli("--works", "W901", "--effort-check", "low", "--tag", "low", "--max-rub", "5")
        self.assertIn("проверка ✓ (2 задач)", out)
        self.assertIn(("check", "low", False), FakeAPI.calls)
        FakeAPI.calls.clear()
        self.run_cli("--works", "W901", "--effort-check", "low", "--tag", "low", "--max-rub", "5")
        self.assertEqual(FakeAPI.calls, [])  # всё из кэша, денег не тратит

    def test_3_settings_guard(self):
        out = self.run_cli("--works", "W901", "--steps", "check", "--effort-check", "high", "--tag", "low")
        self.assertIn("другими настройками", out)
        out = self.run_cli("--works", "W901", "--steps", "transcribe", "--ocr-prompt", "v2")
        self.assertIn("используй --tag", out)

    def test_4_with_key(self):
        out = self.run_cli("--works", "W901", "--steps", "check", "--with-key", "--tag", "nokey_task")
        self.assertIn("сначала запусти", out)  # в основном прогоне ещё нет условия — понятная ошибка
        self.run_cli("--works", "W901", "--steps", "task")
        FakeAPI.calls.clear()
        self.run_cli("--works", "W901", "--steps", "check", "--effort-check", "low", "--with-key", "--tag", "low_key")
        self.assertIn(("check", "low", True), FakeAPI.calls)

    def test_6_double_reading(self):
        FakeAPI.calls.clear()
        self.run_cli("--works", "W901", "--steps", "transcribe", "--ocr-double", "--tag", "dbl")
        self.assertEqual(sorted(c[0] for c in FakeAPI.calls), ["ocr2", "ocr2"])  # первое прочтение не повторяется
        tr = json.load(open(self.tmp / "runs" / "test_model@dbl" / "W901" / "transcription.json", encoding="utf-8"))
        self.assertEqual(tr[0]["disagreements"][0]["a"], "N1: 2+2=5")
        self.assertEqual(tr[0]["disagreements"][0]["b"], "N1: 2+2=6")

    def test_7_check_prompt_v2(self):
        FakeAPI.texts.clear()
        out = self.run_cli("--works", "W901", "--steps", "check", "--effort-check", "low", "--tag", "low",
                           "--check-prompt", "v2")
        self.assertIn("другими настройками", out)  # в папку low (промпт v1) v2 не смешивается
        self.run_cli("--works", "W901", "--steps", "check", "--effort-check", "low", "--tag", "low_v2",
                     "--check-prompt", "v2")
        self.assertTrue(any("КАК НЕ ПРИДИРАТЬСЯ" in t for t in FakeAPI.texts))

    def test_8_preset(self):
        (self.tmp / "config.json").write_text(json.dumps(
            {"final": {"effort_check": "low", "with_key": True, "tag": "low_key", "max_rub": 5}}))
        FakeAPI.calls.clear()
        out = self.run_cli("--preset", "final", "--works", "W901", "--steps", "check")
        self.assertIn("Набор настроек «final»", out)
        self.assertIn("проверка — low", out)
        self.assertEqual(FakeAPI.calls, [])  # тот же набор, что test_4 → из кэша @low_key

    def test_5_report_and_features(self):
        out = self.run_cli("--report")
        self.assertIn("test_model@low", out)
        self.assertIn("совпадение вердиктов             100%", out)
        r = subprocess.run([sys.executable, "features.py"], cwd=self.tmp, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = list(csv.DictReader(open(self.tmp / "data" / "features.csv", encoding="utf-8")))
        low = [x for x in rows if x["run"] == "test_model@low"]
        self.assertEqual(sorted(x["model_wrong"] for x in low), ["0", "0"])
        self.assertEqual({x["key_vs_student"] for x in low}, {"0", "1"})  # 5≠4, 3=3


if __name__ == "__main__":
    unittest.main()
