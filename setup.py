
from os import path
from setuptools import setup, find_packages


VERSION = '1.0'
DESCRIPTION = 'Copy number profiling VAE'

directory = path.dirname(path.abspath(__file__))

with open(path.join(directory, 'requirements.txt')) as f:
    required = f.read().splitlines()

setup(
    name = 'copyVAE',
    version = VERSION,
    description = DESCRIPTION,
    url = 'https://github.com/Lamicc/copyVAE',
    author = 'Chen & Bonet & Kurt',
    packages = find_packages(include = ['copyvae', 'copyvae.*']),
    entry_points = {
        'console_scripts': [
            'copyvae = copyvae.pipeline:main',
        ],
    },
    install_requires = required,
    extras_require = {
        'interactive': ['matplotlib'],
    }
)
