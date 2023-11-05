from spoken.hello import hello


def test_hello(default_person):
    assert hello(default_person) == "Hello, World!"
