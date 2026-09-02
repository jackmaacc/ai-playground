"""LEARNING_PATH.md must stay honest.

A ticked box has to point at real code, and topics we know are not
built must not be ticked. This is the cheapest guard against the roadmap
drifting into wishful thinking.
"""

import re
import sys
import unittest
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = LEARNING_DIR.parent
ROADMAP = REPO_ROOT / "LEARNING_PATH.md"

# Topics that are not implemented. If one of these gets built, tick it in
# the roadmap AND remove it here - deliberately a two-place change.
NOT_YET_BUILT = (
    "pandas",
    "tensor shapes",
    "classification",
    "scikit-learn",
    "backpropagation",
    "tiny neural network",
    "PyTorch",
    "LoRA",
    "CNNs and diffusion",
    "tokenization & embeddings",
    "experiment tracking",
    "model versioning",
    "Hosted APIs",
    "Interview prep",
)


def checkbox_lines():
    for line in ROADMAP.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- [x]") or stripped.startswith("- [ ]"):
            yield stripped


class TickedBoxesCiteRealCode(unittest.TestCase):
    def test_every_ticked_line_has_evidence_that_exists(self):
        problems = []
        ticked = [line for line in checkbox_lines() if line.startswith("- [x]")]
        self.assertTrue(ticked, "expected at least one completed item")
        for line in ticked:
            match = re.search(r"\(evidence: ([^)]+)\)", line)
            if not match:
                problems.append(f"no evidence pointer: {line[:70]}...")
                continue
            for path in (p.strip() for p in match.group(1).split(",")):
                if not (REPO_ROOT / path).exists():
                    problems.append(f"missing evidence file {path!r} on: {line[:60]}...")
        self.assertEqual(problems, [], "\n".join(problems))


class UnbuiltTopicsStayUnticked(unittest.TestCase):
    def test_known_unbuilt_topics_are_not_marked_done(self):
        wrongly_ticked = []
        for line in checkbox_lines():
            if not line.startswith("- [x]"):
                continue
            # Judge the subject of the line (before the em dash), not the
            # explanatory note after it - notes may mention unbuilt things.
            head = line[len("- [x]"):].split("—")[0]
            for topic in NOT_YET_BUILT:
                if topic.lower() in head.lower():
                    wrongly_ticked.append(f"{topic!r} ticked: {line[:70]}...")
        self.assertEqual(wrongly_ticked, [], "\n".join(wrongly_ticked))


class NextLessonSection(unittest.TestCase):
    def test_exists_and_puts_tokenization_first(self):
        text = ROADMAP.read_text(encoding="utf-8")
        self.assertIn("## Next lesson", text)
        section = text.split("## Next lesson", 1)[1]
        first_paragraph = section.strip().split("\n\n", 1)[0]
        self.assertIn("okeniz", first_paragraph)
        self.assertIn("top priority", first_paragraph.lower())

    def test_status_line_is_dated(self):
        text = ROADMAP.read_text(encoding="utf-8")
        self.assertRegex(text, r"Current status \(reviewed \d{4}-\d{2}-\d{2}")


if __name__ == "__main__":
    unittest.main()
