"""
setup.py

Compatibility wrapper for editable installs and older packaging workflows.

Most package metadata lives in pyproject.toml. Keeping this file small helps
avoid conflicts while still allowing commands such as:

    pip install -e .
"""

from setuptools import setup

setup()
