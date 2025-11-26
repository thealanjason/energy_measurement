# Comparative study of different energy measurement tools

## Description


## Installations
0. Clone the project
```bash
git clone https://github.com/thealanjason/energy_measurement.git
```
1. Energy measurement tools
1.1 [Alumet](https://alumet-dev.github.io/user-book/intro.html)
See installation details [here](./alumet/README.md)
2. Dashboard
2.1 [marimo](https://marimo.io/features)
```bash
micromamba env create --file environment.yml
```

## Usage
1. Alumet
How to run energy measurement? See details [here](./alumet/README.md)

Run visualize dashboard.
```bash
cd alumet
marimo run dashboard.py
```

2. PMT

3. Variorum

## Case studies
1. STREAM-Triad benchmark (Memory-bound)

2. GEMM (Compute-bound)