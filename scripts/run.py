"""
Helper script to enable proper importing inside the package.

This file provides a way of running python packages directly from source in VSCode (and
possibly in other ways). With out this, the imports from kyykkaanalysis in
kyykkaanalysis.__main__ fail with ModuleNotFoundError. Relative imports would work
while debugging but fail when running the package after installation.
"""

from kyykkaanalysis import __main__

if __name__ == "__main__":
    __main__.main()
