import pytest

from src.env import ORIGINAL_TEST_PY, clean_code
from src.solver import make_unsolvable, solve

_ns: dict = {}
exec(clean_code(ORIGINAL_TEST_PY), _ns)
verify_solution = _ns["verify_solution"]


@pytest.mark.parametrize(
    "numbers,target",
    [([1, 2, 3], 6), ([44, 19, 35], 98), ([75, 77, 9], 18), ([66, 33, 28, 11], 55), ([4, 6], 1.5)],
)
def test_solutions_pass_the_real_harness(numbers, target):
    expr = solve(numbers, target)
    assert expr is not None
    assert verify_solution(numbers, target, expr)


def test_unary_minus_is_covered():
    """verify_solution's regex admits unary minus, so -(a-b) style answers
    count. 5 - 8 = -3, so target 3 is reachable only via negation or 8 - 5 --
    either way solve must not call it unsolvable."""
    assert solve([5, 8], 3) is not None


def test_known_unsolvable():
    assert solve([2, 2], 7) is None


@pytest.mark.parametrize("numbers,target", [([44, 19, 35], 98), ([66, 33, 28, 11], 55)])
def test_unsolvable_twin_is_unsolvable_nearby_and_deterministic(numbers, target):
    twin = make_unsolvable(numbers, target)
    assert solve(numbers, twin) is None
    assert twin != target and twin > 0 and abs(twin - target) <= 250
    assert make_unsolvable(numbers, target) == twin


def test_unsolvable_twin_never_equals_an_input_number():
    for numbers, target in [([87, 96, 75], 60), ([44, 19, 35], 98), ([10, 20, 30, 40], 25)]:
        for seed in range(20):
            assert make_unsolvable(numbers, target, seed) not in numbers
