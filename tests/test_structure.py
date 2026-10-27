"""Lightweight structural tests that do not require model dependencies."""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

class ProjectStructureTests(unittest.TestCase):
    def test_expected_modules_exist(self):
        expected = [
            "knowledge_aug_gat/__init__.py",
            "knowledge_aug_gat/models/model.py",
            "knowledge_aug_gat/utils/qm9_utils.py",
            "engine/train.py",
            "evaluation/test.py",
            "inference/predict.py",
            "main.py",
        ]
        for relative in expected:
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file())

    def test_python_files_parse(self):
        for path in ROOT.rglob("*.py"):
            if ".git" not in path.parts:
                with self.subTest(path=str(path.relative_to(ROOT))):
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

if __name__ == "__main__":
    unittest.main()
