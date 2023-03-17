from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='openai-experiment',
    version='0.1.0',
    description='OpenAI Experiments',
    long_description=readme,
    author='Michael C. Han',
    url='https://github.com/mhan810/openai-experiments',
    license=license,
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'openai',
        'beautifulsoup4'
    ]
    packages=find_packages(exclude=('tests', 'docs'))
)
