from setuptools import setup, find_packages

setup(
    name='gutcheck',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas', 
        'scikit-learn', 
        'argparse',  # Required for CLI
    ],
    entry_points={
        'console_scripts': [
            'gutcheck=microbiome_bmi_classifier.cli:main',  # CLI entry point
        ],
    },
)
