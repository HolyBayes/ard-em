from setuptools import setup
import io
import os
import re

here = os.path.abspath(os.path.dirname(__file__))


def find_version():
    with io.open(os.path.join(here, '__init__.py'), 'r') as f:
        version_file = f.read()

    # The version line must have the form like following:
    # __version__ = '0.x.x'
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.MULTILINE)
    candidate = str(version_match.group(1))
    assert str.startswith(candidate, '0.') and candidate[2] != '0'
    if version_match:
        return candidate
    raise RuntimeError("Unable to find version string.")


with io.open(os.path.join(here, 'Readme.md')) as f:
    long_description = f.read()

with io.open(os.path.join(here, 'requirements.txt')) as f:
    requirements = f.read().split('\n')

setup(
    name="ard_em",
    version=find_version(),
    description="ARD EM package",
    long_description="ARD (Automatic Relevance Determination) EM algorithm for gaussian mixture separation/clustering\
        with automatic determination of components/clusters number.",
    url='https://github.com/Leensman/ard-em',

    # Author details
    author='Artem Ryzhikov',
    author_email='artemryzhikoff@yandex.ru',

    # Choose your license
    license='Apache-2.0 License',

    # Manually specifying all packages
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7 ',
        'Programming Language :: Python :: 3.4 ',
    ],

    # What does your project relate to?
    keywords='pattern recognition, density reconstruction, cluster analysis,\
clusters number determination, EM algorithm, Bayes Learning, automatic relevance determination',

    # List run-time dependencies here. These will be installed by pip when your
    # project is installed.
    install_requires=requirements,
    py_modules=['ard_em']
)