import click

@click.group()
@click.version_option()
def cli():
    """
    ml_pipeline: a template for building, training and running pytorch models.
    """
