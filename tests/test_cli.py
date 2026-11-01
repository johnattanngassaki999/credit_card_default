import pytest

from credit_card_default.cli import main


def test_cli_requires_a_command():
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code != 0


def test_cli_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "train" in capsys.readouterr().out