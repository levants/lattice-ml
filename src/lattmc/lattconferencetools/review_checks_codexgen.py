"""Focused checks for the reviewed lattice-context implementation."""

from __future__ import annotations

from .paths_codexgen import repository
import ast
import importlib.util
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = repository()
STAGE = ROOT
sys.path.insert(0, str(ROOT))


class ReviewChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls: type[ReviewChecks]) -> None:
        """Prepare shared finite contexts and cached inputs for the checks."""
        path = STAGE / 'src/lattmc/fca/fca_utils_codexgen.py'
        spec = importlib.util.spec_from_file_location('review_fca', path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_empty_extent_closes_to_top(self: ReviewChecks) -> None:
        """Verify empty extent closes to top."""
        fca = self.module.FCA(np.array([[1., 0.], [0., 1.]]))
        concept = fca.map_v(np.array([.5, .5]))
        self.assertEqual(concept.A.size, 0)
        self.assertIn(concept.A.dtype.kind, 'iu')
        np.testing.assert_array_equal(concept.v, [1., 1.])
        np.testing.assert_array_equal(fca.G(concept.v), concept.A)

    def test_closed_intents_and_extents(self: ReviewChecks) -> None:
        """Verify closed intents and extents."""
        fca = self.module.FCA(np.array([[1., 0.], [0., 1.]]))
        for query in [[0., 0.], [1., 0.], [.5, .5]]:
            query = np.array(query)
            intent = fca.FG(query)
            np.testing.assert_array_equal(fca.map_v(query).v, intent)
            np.testing.assert_array_equal(fca.FG(intent), intent)
            np.testing.assert_array_equal(fca.G(intent), fca.G(query))
            self.assertTrue(np.all(query <= intent))

    def test_explicit_lattice_top(self: ReviewChecks) -> None:
        """Verify explicit lattice top."""
        fca = self.module.FCA(
            np.array([[1., 0.], [0., 1.]]), max_val=np.array([2., 3.])
        )
        np.testing.assert_array_equal(fca.F([]), [2., 3.])
        np.testing.assert_array_equal(fca.map_v([.5, .5]).v, [2., 3.])

    def test_zero_source_guard(self: ReviewChecks) -> None:
        """Verify zero source guard."""
        path = STAGE / 'src/lattmc/tc/transcoder_analyzers_codexgen.py'
        tree = ast.parse(path.read_text())
        method = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == '_collect_vals'
        )
        # Execute the actual method, without importing model-loading code.
        method.decorator_list = []
        for argument in method.args.args:
            argument.annotation = None
        method.returns = None
        namespace = {'np': np}
        exec(compile(ast.Module([method], []), str(path), 'exec'), namespace)
        instance = SimpleNamespace(idcs=[0], vs={0: [np.zeros(2)]})
        with self.assertRaisesRegex(ValueError, 'positive source'):
            namespace['_collect_vals'](instance, 0, min_vals=True)

    def test_notebook_pairwise_meets(self: ReviewChecks) -> None:
        """Verify notebook pairwise meets."""
        import nbformat
        count = 0
        for path in (STAGE / 'notebooks').rglob(
            '*_gpt_small_tokens_places_min_acts_codexgen.ipynb'
        ):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type != 'code':
                    continue
                source = cell.source
                if not any(line.lstrip().startswith(('!', '%'))
                           for line in source.splitlines()):
                    ast.parse(source)
                if '# v_3 = concept_an.v_FG' not in source:
                    continue
                namespace = {
                    'np': np, 'layer': 0, 't_idcs': [0, 1],
                    'concept_an': SimpleNamespace(v_FG={
                        0: {0: np.array([4., 3.]), 1: np.array([2., 5.])}
                    }),
                    'meet': np.minimum,
                    'topK': lambda v, k: (v, np.arange(k)),
                    'v_3': np.array([0., 0.]),
                }
                exec(source, namespace)
                np.testing.assert_array_equal(namespace['v_meet'], [2., 3.])
                count += 1
        self.assertEqual(count, 12)


if __name__ == '__main__':
    unittest.main()
