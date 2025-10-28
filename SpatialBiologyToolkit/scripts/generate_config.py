# generate_config_template.py

import yaml
from dataclasses import asdict
from .config_and_utils import (
    GeneralConfig,
    PreprocessConfig,
    CreateMasksConfig,
    SegmentationConfig,
    DenoisingConfig,
    LoggingConfig,
    BasicProcessConfig
)

def generate_config_template(output_file: str = 'config.yaml'):
    """
    Generate a configuration template by extracting configuration parameters
    from the dataclasses defined in config_and_utils.py.
    """
    # Create instances of the dataclasses with default values
    config = {
        'general': asdict(GeneralConfig()),
        'preprocess': asdict(PreprocessConfig()),
        'createmasks': asdict(CreateMasksConfig()),
        'segmentation': asdict(SegmentationConfig()),
        'denoising': asdict(DenoisingConfig()),
        'process': asdict(BasicProcessConfig()),
        'logging': asdict(LoggingConfig()),
    }

    # Write the configuration to a YAML file
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Configuration template generated and saved to '{output_file}'.")

if __name__ == "__main__":
    generate_config_template()
