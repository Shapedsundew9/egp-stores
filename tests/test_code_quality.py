from subprocess import run, PIPE, CompletedProcess


def test_formatting() -> None:
    """Test that the code is formatted correctly."""
    result: CompletedProcess[bytes] = run(["black", "--check", "--diff", "."], stdout=PIPE, stderr=PIPE, check=False)
    assert result.returncode == 0, result.stderr.decode("utf-8")


def test_linting() -> None:
    """Test that the code is linted correctly."""
    result: CompletedProcess[bytes] = run(["pylint", "egp_stores"], stdout=PIPE, stderr=PIPE, check=False)
    assert result.returncode == 0, result.stderr.decode("utf-8")


def test_typing() -> None:
    """Test that the code is typed correctly."""
    result: CompletedProcess[bytes] = run(["pyright"], stdout=PIPE, stderr=PIPE, check=False)
    assert result.returncode == 0, result.stderr.decode("utf-8")
