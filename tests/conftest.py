import pytest


@pytest.fixture
def default_person():
    """The default person to greet"""
    return "World"
