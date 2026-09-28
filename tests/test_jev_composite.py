import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts import jev_score
from scripts.jev_score import (
    build_checks,
    build_questions,
    composite_scores,
    main,
    score_to_attractiveness,
)


def _fake_answer(score=2, confidence=0.8, choice="wait_for_print", noul=0.1):
    return SimpleNamespace(
        score=score,
        confidence=confidence,
        choice=choice,
        noul=noul,
        model_dump=lambda: {"score": score, "confidence": confidence,
                            "choice": choice, "noul": noul},
    )


def _fake_answers():
    answers = {}
    for t in jev_score.DOSSIERS:
        for d in jev_score.DIMENSIONS:
            answers[f"{t}_{d}"] = _fake_answer(score=2, confidence=0.8)
        answers[f"{t}_action"] = _fake_answer(choice="scale_in_on_trigger")
        answers[f"{t}_info_edge"] = _fake_answer(noul=0.15)
        answers[f"{t}_climax"] = _fake_answer(noul=0.2)
    return answers


class TestJevBuilders(unittest.TestCase):
    def test_question_counts(self):
        self.assertEqual(len(build_questions()), 4 * 9)
        self.assertEqual(len(build_checks()), 8)

    def test_question_keys_cover_all_tickers_and_dimensions(self):
        qs = build_questions()
        for t in jev_score.DOSSIERS:
            for d in jev_score.DIMENSIONS:
                self.assertIn(f"{t}_{d}", qs)
            self.assertIn(f"{t}_action", qs)


class TestNormalization(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(score_to_attractiveness(0), 1.0)
        self.assertEqual(score_to_attractiveness(4), 0.0)
        self.assertEqual(score_to_attractiveness(2), 0.5)

    def test_out_of_range_rejected(self):
        for bad in (-1, 5, 99):
            with self.assertRaises(ValueError):
                score_to_attractiveness(bad)


class TestCompositeScores(unittest.TestCase):
    def test_uniform_scores_weight_to_half(self):
        lines = composite_scores(_fake_answers())
        self.assertEqual(len(lines), 4)
        for line in lines:
            self.assertEqual(line["dip_buyer"], 0.5)
            self.assertEqual(line["momentum"], 0.5)
            self.assertEqual(line["action"], "scale_in_on_trigger")
            self.assertEqual(line["invalidate_below"], jev_score.INVALIDATE[line["ticker"]])

    def test_weighting_math(self):
        answers = _fake_answers()
        answers["FSLR_value"] = _fake_answer(score=0, confidence=0.9)
        line = next(l for l in composite_scores(answers) if l["ticker"] == "FSLR")
        self.assertAlmostEqual(line["dip_buyer"], 0.5 + 0.25 * 0.5, places=3)
        self.assertAlmostEqual(line["momentum"], 0.5 + 0.10 * 0.5, places=3)
        self.assertEqual(line["min_conf"], 0.8)


class TestMainWithFakeClient(unittest.TestCase):
    def test_artifacts_written_without_network(self):
        import tempfile

        tmp = tempfile.mkdtemp()
        answers = _fake_answers()

        class FakeClient:
            def __init__(self):
                self.calls = 0

            def system_one(self, state=None, questions=None):
                self.calls += 1
                return SimpleNamespace(model="fake", usage={},
                                       answers=dict(answers))

        main(client=FakeClient(), out_dir=tmp)
        for name in ("jev_scores.json", "jev_checks.json"):
            payload = json.loads((Path(tmp) / name).read_text())
            self.assertEqual(payload["model"], "fake")
            self.assertTrue(payload["answers"])


if __name__ == "__main__":
    unittest.main()
