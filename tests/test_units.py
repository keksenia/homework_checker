"""Быстрые тесты без сети: сравнение ответов, номера задач, отбор условий, метрика «автоисправления».
Запуск из папки проекта:  python3 -m unittest discover tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.argv = sys.argv[:1]
import check  # noqa: E402
import features  # noqa: E402
import ocr_eval  # noqa: E402


class AnswersMatch(unittest.TestCase):
    CASES = [
        (r"\frac{\sqrt2}{2}", "√2/2", "number", 1), (r"\frac{1}{\sqrt3}", "√3/3", "number", 1),
        (r"\frac{3}{4}\pi", "3π/4", "number", 1), (r"\frac12", "0,5", "number", 1),
        (r"\tfrac{1}{2}", "0.5", "number", 1), (r"x_{1,2}=\pm3", "-3; 3", "set", 1),
        ("—3", "-3", "number", 1), (r"2^{10}", "1024", "number", 1), (r"10^{-2}", "0,01", "number", 1),
        ("x ≤ 1", "(-∞; 1]", "interval", None), ("(x-1)(x+2)", "x^2+x-2", "expression", None),
        ("(0;1);(2;3)", "(0;2);(1;3)", "pair", 0), ("(1; 2); (-1; -2)", "(-1;-2),(1;2)", "pair", 1),
        ("1, 10", "1; 10", "set", 1), ("2,5", "2.5", "number", 1), ("-3/5", "-0,6", "number", 1),
        ("2√3", "√12", "number", 1), (r"$\left(-\infty; -3\right) \cup (1; +\infty)$", "(-∞; -3) ∪ (1; +∞)", "interval", 1),
        ("(-∞; 4]", "(-∞; 4)", "interval", 0), ("88", "38", "number", 0), ("19√3", "19", "number", 0),
        (r"-\frac{19\pi}{4}", "-19π/4", "number", 1), ("0,75", "3/4", "number", 1), ("—", "5", "proof", None),
    ]

    def test_cases(self):
        for a, b, kind, exp in self.CASES:
            with self.subTest(a=a, b=b, kind=kind):
                self.assertEqual(features.answers_match(a, b, kind), exp)


class TaskNumbers(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(check.norm_task("№13 а)"), check.norm_task("13а"))
        self.assertEqual(check.norm_task("B1-2.1"), check.norm_task("В1-2.1"))  # латиница ~ кириллица

    def test_hit(self):
        n = check.norm_task
        self.assertTrue(check.task_hit(n("13"), n("13а")))
        self.assertTrue(check.task_hit(n("5"), n("Б1-5")))
        self.assertTrue(check.task_hit(n("Б2-8"), n("15-Б2-8")))
        self.assertFalse(check.task_hit(n("1"), n("13")))
        self.assertEqual(n("Задание 3"), n("3"))
        self.assertEqual(n("задача №17"), n("17"))

    def test_select_tasks_full_coverage(self):
        tasks = [{"task_no": "13"}, {"task_no": "14"}, {"task_no": "15"}]
        self.assertEqual(check.select_tasks(tasks, ["13а", "13б"]), [{"task_no": "13"}])
        # один номер не сопоставился → отдаём всё условие, чтобы не проверять вслепую
        self.assertEqual(check.select_tasks(tasks, ["13", "99"]), tasks)

    def test_pdf_pages(self):
        texts = ["1. Первая задача\n2. Вторая", "25.1. Задача с подпунктом", "7. Седьмая задача"]
        self.assertEqual(check.pdf_needed_pages(texts, ["25.1а"]), [1])
        self.assertEqual(check.pdf_needed_pages(texts, ["7"]), [2])
        self.assertIsNone(check.pdf_needed_pages(texts, ["42"]))  # не найден → все страницы


class Blocks(unittest.TestCase):
    TEXTS = ["Блок 1. ФИПИ\n1. Задача\n101. Сторона квадрата", "99. Ещё\nБлок 2. Расширенная\n101. Биссектрисы"]

    def test_block_prefix(self):
        heads, start, first = check.pdf_blocks(self.TEXTS)
        self.assertEqual(check.with_block("101", self.TEXTS[0], heads[0], start[0], first), "Б1-101")
        self.assertEqual(check.with_block("99", self.TEXTS[1], heads[1], start[1], first), "Б1-99")  # до заголовка
        self.assertEqual(check.with_block("101", self.TEXTS[1], heads[1], start[1], first), "Б2-101")
        self.assertEqual(check.with_block("Б2-5", self.TEXTS[1], heads[1], start[1], first), "Б2-5")
        self.assertIsNone(check.pdf_blocks(["1. без блоков"]))

    def test_select_prefers_block1_for_plain_number(self):
        tasks = [{"task_no": "Б1-101", "text": "квадрат"}, {"task_no": "Б2-101", "text": "трапеция"}]
        self.assertEqual([t["text"] for t in check.select_tasks(tasks, ["101"])], ["квадрат"])
        self.assertEqual([t["text"] for t in check.select_tasks(tasks, ["Б2-101"])], ["трапеция"])

    def test_missing(self):
        tasks = [{"task_no": "Б1-1"}, {"task_no": "5"}]
        self.assertEqual(check.missing_tasks(tasks, ["1", "Б1-5", "7"]), ["7"])


class OcrMetric(unittest.TestCase):
    def test_student_errors(self):
        lines = ["44+44=38", "180-66=24", "2x^2-25=0", "x = (-5 ± 7)/2 = 1", "~~3+3=7~~", "12:4=3"]
        self.assertEqual(ocr_eval.student_arith_errors(lines), ["44+44=38", "180-66=24"])


def _row(work, task, label, pred):
    return {"work": work, "task_no": task, "label": label, "pred": pred, "human_missed": False, "split": "dev"}


class Stats(unittest.TestCase):
    def test_ci_contains_point_and_is_deterministic(self):
        rows = [_row(f"W{w:03d}", str(t), int(t % 3 != 0), int(t % 3 != 0) if (w + t) % 7 else 1 - int(t % 3 != 0))
                for w in range(20) for t in range(10)]
        m = check.metrics(rows)
        self.assertIn("[", m["совпадение вердиктов"])
        self.assertEqual(check.bootstrap_ci(rows), check.bootstrap_ci(rows))
        agree = sum(r["pred"] == r["label"] for r in rows) / len(rows)
        lo, hi = check.bootstrap_ci(rows)[3]
        self.assertTrue(lo <= agree <= hi)

    def test_paired(self):
        a = [_row(f"W{w}", str(t), 1, 1) for w in range(10) for t in range(5)]
        b = [dict(r, pred=0) if r["task_no"] == "0" else r for r in a]  # B ошибается в 10 задачах
        c = check.paired_compare(a, b)
        self.assertEqual((c["прав только A"], c["прав только B"], c["задач"]), (10, 0, 50))
        self.assertLess(c["p (Мак-Немар)"], 0.01)
        self.assertLess(c["ДИ разницы B−A"][1], 0)


class TestLock(unittest.TestCase):
    def test_lock_blocks_changes(self):
        import json
        import tempfile
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "f.txt").write_text("a")
            old = check.ROOT, check.TEST_LOCK, check.REPORTS
            check.ROOT, check.REPORTS, check.TEST_LOCK = d, d, d / "lock.json"
            try:
                settings = {k: None for k in check.RUN_SETTINGS}
                args = SimpleNamespace(**settings)
                with self.assertRaises(SystemExit):  # заморозки нет
                    check.verify_test_lock(args)
                json.dump({"frozen_at": "t", "settings": settings, "files": {"f.txt": check.sha256(d / "f.txt")}},
                          open(check.TEST_LOCK, "w"))
                check.verify_test_lock(args)  # всё совпадает — проходит и пишет журнал
                self.assertTrue((d / check.TEST_RUNS_LOG).exists())
                args.effort_check = "high"
                with self.assertRaises(SystemExit):  # другие настройки
                    check.verify_test_lock(args)
                args.effort_check = None
                (d / "f.txt").write_text("b")
                with self.assertRaises(SystemExit):  # изменился файл
                    check.verify_test_lock(args)
            finally:
                check.ROOT, check.TEST_LOCK, check.REPORTS = old


class PromptsV3(unittest.TestCase):
    def test_v3_prompts_format(self):
        p = check.CHECK_PROMPTS["v3"].format(numbering=check.NUMBERING, scope="")
        self.assertIn("check_next", p)
        self.assertIn("АДЕКВАТНОСТЬ", p)
        self.assertIn('"results"', p)  # JSON-схема не сломана подстановкой
        o = check.OCR_PROMPTS["v3"]
        self.assertIn("перекладиной", o)
        self.assertTrue(o.rstrip().endswith('"legibility": "good|medium|poor"}'))
        self.assertNotEqual(check.OCR_PROMPTS["v3"], check.OCR_PROMPTS["v1"])


class ParseJson(unittest.TestCase):
    def test_robust(self):
        self.assertEqual(check.parse_json('[{"a": 1}, {"b": 2}]'), {"results": [{"a": 1}, {"b": 2}]})
        self.assertEqual(check.parse_json('{"results": []} мусор'), {"results": []})
        r = check.parse_json('{"results": [{"a": 1}, {"b": "x ] }"}\u7f38')  # оборванный конец + мусорный символ
        self.assertEqual(r["results"], [{"a": 1}, {"b": "x ] }"}])
        self.assertTrue(r["_repaired"])
        with self.assertRaises(ValueError):
            check.parse_json("нет json")


class TrustFeatures(unittest.TestCase):
    def test_build(self):
        import pandas as pd
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
        import trust_features as tf
        w = pd.DataFrame({"g_low_key__pred_error": [1, 1, 0, 0, None], "g_low_key__key_vs_student": [1, 0, 0, None, 1],
                          "ds_high_img__pred_error": [0, 1, 0, 1, 0], "legibility_mean": [2, 1, 0, 2, 2],
                          "g_low_key__n_err_substantive": [0, 2, 0, 0, 0]})
        f = tf.build(w)
        self.assertEqual(list(f["key_conflict"]), [1, 0, 1, 0, 0])      # «ошибка» при ответе = ключу; «верно» при ≠
        self.assertEqual(list(f["alarm_no_errors"]), [1, 0, 0, 0, 0])   # тревога без названных ошибок
        self.assertEqual(list(f["second_disagree"]), [1, 0, 0, 1, 0])   # нет вердикта Gemini — несогласия нет
        self.assertEqual(tf.FEATURES_V2, ["g_low_key__pred_error", "key_conflict", "legibility_mean", "alarm_no_errors"])
        self.assertEqual(tf.FEATURES, tf.FEATURES_V2 + ["near_key", "key_missing"])
        self.assertEqual(list(f["key_missing"]), [0, 0, 0, 1, 0])        # ответ не с чем сверить
        f2 = tf.build(w.drop(columns=["ds_high_img__pred_error"]))       # без прогона DeepSeek строится
        self.assertTrue(f2["second_disagree"].eq(0).all())

    def test_near_key(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
        import trust_features as tf
        yes = [("731", "131"), ("3231", "231"), ("0,2", "0,02"), ("15450", "15750"),   # одна цифра
               ("0,25", "4"), ("94", "86"), ("2,25", "1,5"), ("115", "65")]            # 1/x, 180 − x, x², 180 − x
        no = [("12", "12"), ("28", "8,5"), ("-3; 2", "(-∞; 4)"), ("", "5"), ("7", "")]
        for a, b in yes:
            self.assertTrue(tf.near_key(a, b), (a, b))
        for a, b in no:
            self.assertFalse(tf.near_key(a, b), (a, b))

    def test_metrics_count_undecided_as_mismatch(self):
        rows = [_row("W1", "1", 1, 1), _row("W1", "2", 0, 0), _row("W1", "3", 1, None)]
        m = check.metrics(rows, ci=False)
        self.assertEqual(m["совпадение вердиктов"], "67%")   # задача без вердикта — несовпадение



class TestAnswers(unittest.TestCase):
    def test_answers_extract(self):
        import answers
        pages = [{"page": 1, "lines": ["(I)", "1) y = x^3 - 27x", "x = \\pm 3", "Отв: -3", "2) 0,81 - 0,56 = 0,25",
                                       "3) Ответ дайте в градусах", "№4", "а) 2x = 4", "Ответ: 2", "б) x > 1", "Ответ: (1; +\\infty)"]}]
        r = answers.extract(pages, ["1", "1.1", "2", "3", "4а", "4б", "7"])
        self.assertEqual(r["1"]["answer"], "-3")
        self.assertEqual(r["1.1"]["answer"], "-3")          # раздел (I) → 1.1
        self.assertEqual(r["2"]["answer"], "0,25")          # короткий блок: правая часть после «=»
        self.assertEqual(r["3"]["source"], "short")          # «Ответ дайте…» — не строка ответа
        self.assertEqual(r["4а"]["answer"], "2")
        self.assertEqual(r["4б"]["answer"], "(1; +\\infty)")
        self.assertEqual(r["7"]["source"], "no_block")


class TestPolicy(unittest.TestCase):
    def test_policy_modes(self):
        import policy
        pol = policy.load_policy()
        d = lambda sit, mode, **kw: policy.decide([{"situation": sit, **kw}], mode, pol)[0]
        self.assertEqual(d("1.1", "exam"), 0)          # платёж вместо суммы — ошибка на экзамене
        self.assertEqual(d("1.1", "training"), 1)      # и замечание на тренировке
        self.assertEqual(d("2.2", "exam"), 1)          # неизвлечённый корень — замечание везде
        self.assertEqual(d("G.calc", "training"), 0)
        self.assertEqual(d("G.none", "exam"), 1)
        v, cur, _ = policy.decide([{"situation": "other"}], "training", pol)
        self.assertEqual((v, cur), (0, True))          # неизвестное — ошибка + куратор
        v, cur, _ = policy.decide([{"situation": "8.1", "reading_doubt": True}], "training", pol, doubt="remark")
        self.assertEqual((v, cur), (1, True))          # сомнение в прочтении — к куратору, не ошибка
        v, cur, rem = policy.decide([{"situation": "G.calc", "consequence": "fixed_later"}], "exam", pol)
        self.assertEqual((v, cur, rem), (1, True, ["3.1"]))  # дальше верное значение → описка, замечание + куратор
        self.assertEqual(d("G.calc", "exam", consequence="propagated"), 0)
        self.assertEqual(d("4.3", "exam"), 1)          # решение 30.09: цепочка «=» — замечание

    def test_v4_prompt_formats(self):
        p = check.CHECK_PROMPTS["v4"].format(numbering=check.NUMBERING, scope="x")
        self.assertIn("СПРАВОЧНИК СИТУАЦИЙ", p)
        self.assertIn("1.12 —", p)
        self.assertIn('"deviations"', p)

class TestTagger(unittest.TestCase):
    def test_prompt_filled(self):
        import tagger
        item = {"id": "W000|1", "mode": "exam", "condition": "x", "key": "1", "student_answer": "2",
                "errors": [{"type": "calc", "step": "1+1=3", "explanation": "{не формат}"}], "comment": ""}
        p = tagger.prompt_for([item])
        self.assertNotIn("<<", p)
        self.assertIn("3.11 —", p)
        self.assertIn("W000|1", p)
        self.assertIn("{не формат}", p)


if __name__ == "__main__":
    unittest.main()


class ScopeFile(unittest.TestCase):
    def test_load_and_args(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("work_id,task_no\nW900,5\nW900,6а\nW900,5\nW901, Б1-2 \n")
        sc = check.load_scope_file(f.name)
        self.assertEqual(sc, {"W900": ["5", "6а"], "W901": ["Б1-2"]})
        _, args = check.parse_args(["--works", "W900", "--scope-file", f.name])
        self.assertEqual(args.scope_file, f.name)
        self.assertIn("scope_file", check.RUN_SETTINGS)  # смена режима scope ломает заморозку test

    def test_scope_file_refused_on_test(self):
        with self.assertRaises(SystemExit):
            check.main(["--works", "all", "--split", "test", "--scope-file", "x.csv"])
