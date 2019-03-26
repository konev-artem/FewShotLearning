from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fewshot',
    version='0.0.1',
    description='FewShotLearning framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='The MIT License',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
    ],

    packages=find_packages(),

    keywords='cv',
)
