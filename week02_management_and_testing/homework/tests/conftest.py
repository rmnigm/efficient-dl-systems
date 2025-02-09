from utils import set_seed

import pytest

# idea from https://github.com/pytest-dev/pytest/issues/667#issuecomment-112206152
@pytest.fixture(scope='session', autouse=True)
def seed():
    set_seed(123)
