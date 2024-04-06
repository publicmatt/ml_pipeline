from ml_pipeline import config
from ml_pipeline.model.cnn import VGG11

def test_in_channels():
    assert config.model.name == 'vgg11' 

