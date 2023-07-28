import typer

file_in_option = typer.Option(
    exists=True,
    file_okay=True,
    dir_okay=False,
    writable=False,
    readable=True,
    resolve_path=True,
)
file_out_option = typer.Option(
    exists=False,
    file_okay=True,
    dir_okay=False,
    writable=True,
    readable=True,
    resolve_path=True,
)
dir_in_option = typer.Option(
    exists=True,
    file_okay=False,
    dir_okay=True,
    writable=False,
    readable=True,
    resolve_path=True,
)
