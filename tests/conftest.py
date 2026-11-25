import pytest

from litmine.config import load_config
from litmine.demo_data import demo_lexicon_path, demo_works
from litmine.workbench import Workbench


@pytest.fixture(scope="session")
def works():
    return demo_works()[0]


@pytest.fixture(scope="session")
def wb(works):
    return Workbench(works, load_config(), demo_lexicon_path(), label="demo")
