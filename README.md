# shogiraffe

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Shogiraffe is a shogi AI inspired by Matthew Lai's thesis called Giraffe (link).

## Documentation
The documentation is coming soon !

## Install

`pip install git+https://github.com/Theo-Cheynel/shogiraffe`

Or clone the repo and install locally with pip :
```
git clone git@github.com:Theo-Cheynel/shogiraffe.git
cd shogiraffe
pip install .
```

## Development
To ensure that you follow the development workflow, please install pre-commit :
```
pip install pre-commit
```

And setup the pre-commit hooks :

```
pre-commit install
```
Now, every time you commit a file, it will run the Black formatter before committing.
It will also run several hooks to ensure that your code is clean, like `isort` or `debug-statements`.

## Continuous Integration

The package contains a `tests` folder, with both unit tests and integration tests.

The unit tests are run on an AWS unit test server, using the Python framework `unittest`. Unit tests are written for each function, to ensure that they work as expected. Integration tests are written for scenes as a whole, and ensure that the functions interact with one another without any error.
