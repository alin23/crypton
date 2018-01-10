import pathlib
from io import open  # pylint: disable=redefined-builtin

from setuptools import setup, find_packages

CONFIGDIR = pathlib.Path.home() / '.config' / 'crypton'
CONFIGDIR.mkdir(parents=True, exist_ok=True)

with open('crypton/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()  # pylint: disable=invalid-name

REQUIRES = [
    "addict",
    "ccxt",
    "fbprophet",
    "fire",
    "keras",
    "kick",
    "matplotlib",
    "numpy",
    "pandas",
    "python-dateutil",
    "tensorflow"
    "uvloop",
]

setup(
    name='crypton',
    version=version,
    description='',
    long_description=readme,
    author='Alin Panaitiu',
    author_email='alin.p32@gmail.com',
    maintainer='Alin Panaitiu',
    maintainer_email='alin.p32@gmail.com',
    url='https://github.com/alin23/crypton',
    license='MIT/Apache-2.0',
    keywords=[
        '',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    install_requires=REQUIRES,
    tests_require=['coverage', 'pytest'],
    packages=find_packages(),
    package_data={'crypton': ['config/config.toml']},
    entry_points={'console_scripts': ['crypton = crypton.crypton:main']},
    data_files=[(str(CONFIGDIR), ['crypton/config/config.toml'])],
)
