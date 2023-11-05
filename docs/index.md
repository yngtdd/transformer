# Transformer

Welcome to documentation for the `Transformer` project.

---

## Installation

```console
hatch build
```

## Documentation

Docs are built alongside our source code and hosted [here](https://yngtdd.github.io/transformer/).
When offline, a local version of the same documenation can be served in the browser using


```console
hatch run docs:serve
```

## Tests

The test suite can be run, with coverage, using

```console
hatch run cov
```

## Development and Exploration

A Jupyter Lab instance can be launched from any of the project directories using the following command.
This will launch Jupyter Lab within the `examples/notebooks` directory.


```console
hatch run dev:lab
```

If you would like something more lightweight, iPython is also included

```console
hatch run dev:ipython
```
