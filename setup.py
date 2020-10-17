import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='stacked-ensemble-model-ezeray',
    version='0.0.1',
    author='Ezequiel Raigorodsky',
    author_email='ezequielraigorodsky@gmail.com',
    description=(
        'Implementation for a stacked ensemble model, which includes option '
        'to incorporate a preprocessing step, the base models and a final '
        'metamodel.'
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ezeray/stacked-ensemble-model',
    packages=setuptools.find_packages(),
        classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
