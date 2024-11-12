"""CLI for Skore."""

import pathlib
from importlib.metadata import version

import click

from skore.cli.color_format import ColorHelpFormatter
from skore.cli.create_project import __create
from skore.cli.launch_dashboard import __launch
from skore.cli.quickstart_command import __quickstart


click.Context.formatter_class = ColorHelpFormatter


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option()
def main():
    """Command-line interface for skore."""
    pass


@main.command()
@click.argument("project_name", default="project", required=False)
@click.option(
    "--port",
    type=int,
    default=22140,
    help="Port to run the server on (default: 22140)",
)
@click.option(
    "--open-browser/--no-open-browser",
    default=True,
    help=(
        "Whether to automatically open a browser tab showing the web UI "
        "(default: True)"
    ),
)
def launch(project_name, port, open_browser):
    """Launch the skore UI server."""
    __launch(project_name=project_name, port=port, open_browser=open_browser)


@main.command()
@click.argument("project_name", default="project", required=False)
@click.option(
    "--working-dir",
    type=click.Path(path_type=pathlib.Path),
    help="The directory relative to which the project name will be interpreted",
)
def create(project_name, working_dir):
    """Create a new skore project."""
    __create(project_name=project_name, working_dir=working_dir)


@main.command()
def quickstart():
    """Create a "project.skore" file and start the UI."""
    __quickstart()


def start():
    main()
