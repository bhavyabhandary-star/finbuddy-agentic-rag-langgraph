"""Loads .env before any test runs, so ANTHROPIC_API_KEY etc. are available to
integration tests without needing to export them in the shell every time.
"""
from dotenv import load_dotenv

load_dotenv()
