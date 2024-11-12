"""Entry point to Skore CLI."""

import sys

from skore.cli.cli import start

if __name__ == "__main__":
    import rich.traceback

    # Display error tracebacks with Rich
    rich.traceback.install(show_locals=True)

    start(sys.argv[1:])
