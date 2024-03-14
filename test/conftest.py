# conftest.py
import pytest
import os
from dotenv import load_dotenv
from pathlib import Path


@pytest.fixture(autouse=True)
def load_env():
    # Set up your environment variables here
    env = Path(__file__).parent / ".env.test"
    if not load_dotenv(env):
        raise RuntimeError(".env not loaded")
    # os.environ['MY_ENV_VAR'] = 'some_value'
    # You can add more setup code here if needed

    yield

    # Optional: Cleanup code after test (if needed)
    # e.g., unset environment variables if they should not persist after test
