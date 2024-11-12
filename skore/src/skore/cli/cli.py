"""CLI for Skore."""

import argparse
import pathlib
from importlib.metadata import version

import click

from skore.cli.create_project import __create
from skore.cli.launch_dashboard import __launch
from skore.cli.quickstart_command import __quickstart


# def cli(args: list[str]):
#     """CLI for Skore."""
#     parser = argparse.ArgumentParser(prog="skore")

#     parser.add_argument(
#         "--version", action="version", version=f"%(prog)s {version('skore')}"
#     )

#     subparsers = parser.add_subparsers(dest="subcommand")

#     parser_launch = subparsers.add_parser("launch", help="Launch the web UI")
#     parser_launch.add_argument(
#         "project_name",
#         help="the name or path of the project to open",
#     )
#     parser_launch.add_argument(
#         "--port",
#         type=int,
#         help="the port at which to bind the UI server (default: %(default)s)",
#         default=22140,
#     )
#     parser_launch.add_argument(
#         "--open-browser",
#         action=argparse.BooleanOptionalAction,
#         help=(
#             "whether to automatically open a browser tab showing the web UI "
#             "(default: %(default)s)"
#         ),
#         default=True,
#     )

#     parser_create = subparsers.add_parser("create", help="Create a project")
#     parser_create.add_argument(
#         "project_name",
#         nargs="?",
#         help="the name or path of the project to create (default: %(default)s)",
#         default="project",
#     )
#     parser_create.add_argument(
#         "--working-dir",
#         type=pathlib.Path,
#         help=(
#             "the directory relative to which the project name will be interpreted; "
#             "default is the current working directory (mostly used for testing)"
#         ),
#         default=None,
#     )

#     subparsers.add_parser(
#         "quickstart", help='Create a "project.skore" file and start the UI'
#     )

#     parsed_args: argparse.Namespace = parser.parse_args(args)

#     if parsed_args.subcommand == "launch":
#         __launch(
#             project_name=parsed_args.project_name,
#             port=parsed_args.port,
#             open_browser=parsed_args.open_browser,
#         )
#     elif parsed_args.subcommand == "create":
#         __create(
#             project_name=parsed_args.project_name,
#             working_dir=parsed_args.working_dir,
#         )
#     elif parsed_args.subcommand == "quickstart":
#         __quickstart()
#     else:
#         parser.print_help()

# import click


@click.group()
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
    help="Whether to automatically open a browser tab showing the web UI (default: True)",
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
