"""Exhaustive checks of saturation and compressed one-pair bonds."""

from __future__ import annotations
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator

import itertools
import json
import unittest

from .bond_experiments_codexgen import one_pair, saturate
from .paths_codexgen import OUT


def le(a: int, b: int) -> bool:
    """Test inclusion between bit-encoded Boolean-lattice elements."""
    return a & b == a


def join(values: Iterable[int]) -> int:
    """Compute the Boolean join of bit-encoded elements."""
    result = 0
    for value in values:
        result |= value
    return result


def closure_families() -> Iterator[Callable[[int], int]]:
    """Enumerate closure operators on the two-atom Boolean lattice."""
    for mask in range(1 << 4):
        closed = [a for a in range(4) if mask & (1 << a)]
        if 3 not in closed:
            continue
        if any(a & b not in closed for a in closed for b in closed):
            continue
        def c(a: int, closed: list[int] = closed) -> int:
            """Close a bit-encoded element in the selected closure system."""
            result = 3
            for b in closed:
                if le(a, b):
                    result &= b
            return result
        yield c


def is_bond(
    relation: set[tuple[int, int]],
    cx: Callable[[int], int],
    dv: Callable[[int], int],
) -> bool:
    """Check that all relation fibers are closed principal ideals."""
    for a in range(4):
        row = [v for b, v in relation if a == b]
        maximum = join(row)
        if set(row) != {v for v in range(4) if le(v, maximum)}:
            return False
        if dv(maximum) != maximum:
            return False
    for v in range(4):
        column = [a for a, w in relation if v == w]
        maximum = join(column)
        if set(column) != {a for a in range(4) if le(a, maximum)}:
            return False
        if cx(maximum) != maximum:
            return False
    return True


class BondChecks(unittest.TestCase):
    def test_one_pair_all_boolean_two_closures(self: BondChecks) -> None:
        """Verify one pair all boolean two closures."""
        count = 0
        for cx, dv in itertools.product(closure_families(), repeat=2):
            for a, v in itertools.product(range(4), repeat=2):
                seed = {(a, v)}
                actual, history = saturate(
                    seed, range(4), range(4), le, le, join, join, cx, dv
                )
                expected = one_pair(
                    a, v, range(4), range(4), le, le, cx, dv, 0, 0
                )
                self.assertEqual(actual, expected)
                self.assertTrue(is_bond(actual, cx, dv))
                self.assertLessEqual(len(history)-1, 16-len(seed))
                count += 1
        print('One-pair Boolean-lattice instances:', count)

    def test_all_relations_against_intersection_of_bonds(
        self: BondChecks,
    ) -> None:
        # Exhaust all 2^9 seeds on a three-element chain, independently
        # enumerate bonds by their row maxima, and intersect the supersets.
        """Verify all relations against intersection of bonds."""
        xs = range(3)
        pairs = list(itertools.product(xs, repeat=2))
        leq = lambda a, b: a <= b
        sup = lambda values: max(values, default=0)
        cx = lambda a: max(1, a)
        dv = lambda v: v
        bonds = []
        for maxima in itertools.product(xs, repeat=3):
            candidate = {(a, v) for a, v in pairs if v <= maxima[a]}
            valid = True
            for v in xs:
                column = {a for a, w in candidate if w == v}
                top = sup(column)
                valid &= column == set(range(top+1)) and cx(top) == top
            if valid:
                bonds.append(candidate)
        for mask in range(1 << len(pairs)):
            seed = {p for i, p in enumerate(pairs) if mask & (1 << i)}
            expected = set.intersection(*(b for b in bonds if seed <= b))
            actual, _ = saturate(
                seed, xs, xs, leq, leq, sup, sup, cx, dv
            )
            self.assertEqual(actual, expected)
        print('Exhaustive arbitrary-seed chain instances:', 512)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BondChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    output = OUT
    output.mkdir(exist_ok=True)
    (output / 'tests.json').write_text(json.dumps(dict(
        tests=result.testsRun, success=result.wasSuccessful(),
        one_pair_instances=49*16, arbitrary_seed_instances=512,
    ), indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
