"""Declarative, spatially aligned publication figures.

Recipes import without scientific dependencies. Dataset/rendering imports are lazy.
"""
from .specs import (Figure, Panel, Crop, Style, ScaleBar, Scale, Channel, IMC, Image,
                    Populations, Values, LabelMask, Value, Condition, Intensities, obs, var)

__all__ = ['Dataset', 'Figure', 'Panel', 'Crop', 'Style', 'ScaleBar', 'Scale', 'Channel',
           'IMC', 'Image', 'Populations', 'Values', 'LabelMask', 'Value', 'Condition', 'Intensities', 'obs', 'var']


def __getattr__(name):
    if name == 'Dataset':
        from .sources import Dataset
        return Dataset
    raise AttributeError(name)
