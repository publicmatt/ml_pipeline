import click

@click.group()
@click.version_option()
def cli():
    """
    ml_pipeline: a template for building, training and running pytorch models.
    """


@cli.command("train")
def train():
    """run the training pipeline with train data"""
    from ml_pipeline.training.pipeline import run
    run()

@cli.command("evaluate")
def evaluate():
    """run the training pipeline with test data"""
    from ml_pipeline.training.pipeline import run
    run(evaluate=True)
