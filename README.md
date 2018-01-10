# crypton

-----

**Table of Contents**

* [Installation](#installation)
* [License](#license)

## Installation

crypton is distributed on [PyPI](https://pypi.org) as a universal
wheel and is available on Linux/macOS and Windows and supports
Python 3.6+.

```bash
$ pip install crypton
```

## Forecasting

### Using Keras neural networks (useless right now)

```python
crypton --pair XVG,BTC --predictor keras - forecast --learn-from-date='15 December 2017'
```

### Using Facebook Prophet

```python
crypton --pair XVG,BTC --predictor prophet - forecast --learn-from-date='15 December 2017'
```

## License

crypton is distributed under the terms of both

- [MIT License](https://choosealicense.com/licenses/mit)
- [Apache License, Version 2.0](https://choosealicense.com/licenses/apache-2.0)

at your option.
