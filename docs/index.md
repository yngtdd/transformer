# Transformer

Welcome to documentation for the `Transformer` project. 

---

## Installation

This project uses [`rye`](https://rye-up.com/guide/), a Python package management system
built to keeps things development simple and reliable.

Setting up your entire toolchain, creating a virtual environment, and installing `throughput` is
as simple as

```console
rye sync
```

## Documentation

Docs are built alongside our source code and hosted [here](https://yngtdd.github.io/transformer/).
When offline, a local version of the same documenation can be served in the browser using


```console
rye run docs
```

## Tests

The test suite can be run, with coverage, using

```console
rye run test
```

## Benchmarks

The benchmark suite can be run using

```console
rye run bench
```
