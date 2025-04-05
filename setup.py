from setuptools import setup, find_packages

setup(
    name='gutcheck',
    version='0.1.0',
    description='A microbiome-based BMI classification tool using OTU tables and metadata.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/Gouri117/GutCheck',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
    ],
    entry_points={
        'console_scripts': [
            'gutcheck=microbiome_bmi_classifier.cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)

