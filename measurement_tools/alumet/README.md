# Alumet for (Process Specific) Energy Measurement

Alumet (https://alumet.dev/) is a tool that allows measurement of data from various sources, transform them if necessary, and output them to various endpoints. 

## Installation

Here I proceed with installing Alumet from source. For this I needed to perform the following:

### Install Rust Tool Chain

The install instructions were available at https://rust-lang.org/tools/install/. 
On my Debian 13 linux machine, to I ran the command
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Everything went well!


### Install Alumet through `cargo`

The instructions were available at https://alumet-dev.github.io/user-book/start/install.html#option-4-installing-from-source

The specific command I used was: 
```bash
cargo install --git https://github.com/alumet-dev/alumet.git alumet-agent
```

This did not work on the first try, because the `pkg-config` and `OpenSSL` package were missing on my Debian 13 machine. I was able to fix it by installing them both with the commands

```bash
sudo apt install pkg-config
sudo apt install OpenSSL
```

However, on Ubuntu machines, these packages should be available by default. If not, they can be installed using the above commands.


## Going Hands-on with Alumet

### Basics

From going through the website (https://alumet.dev) and user guide (https://alumet-dev.github.io/user-book/), I gathered that Alumet has three key components:

1. Sources - These are points from where measurement data (timeserie) enters the pipeline 
2. Transforms - These are formulas for transformations (through algebraic combination) of the measured values into derived quantities
3. Outputs - These are the endpoints where both raw (from source) and transformed data can be streamed (or written) to.

There can be one or more of Sources, Tranforms and Outputs in a given pipeline. Each of these are provided through a plugin ecosystem. For example, `procfs` is a Source giving information of all the process on a machine, `csv` an Output format to write the data into.


### My first Alumet experiment

To actually use alumet, we need to run an Agent that uses a set of plugins, either throught the command line or through a configuration file. 

In the following command, we run run the agent with the `procfs` and `csv` plugins, by specifying throught the command line. 

Folder: `experiments/00_getting_started`

```bash
alumet-agent --plugins procfs,csv exec python3 -- -c "print(1+1)"
```

The dummy program we want to specifically look at is `python3 -c "print(1+1)"

Here by using `exec` command, the agent is first started, the command is executed, and when the command exits, the agent is stopped.


Since we did not provide a `alumet-config.toml` for the plugins, the agent creates one for us, based on the plugins we specified. The agent also writes the output to `alumet-output.csv` file. The terminal output can be found in `alumet-agent.log`.

### Experiments

#### 1. Fiddling with the `rapl` plugin

Folder: `experiments/01_rapl`

Command: 
```bash
alumet-agent --config=alumet-config-rapl.toml exec python3 ../../../case_studies/test_program.py 2> alumet-agent-rapl.log
```
Output: `alumet-output-rapl.csv`
Logfile: `alumet-agent-rapl.log`

#### 2. Fiddling with the `rapl` and `perf` plugin

Folder: `experiments/02_rapl_perf`

Command: 
```bash
alumet-agent --config=alumet-config-rapl+perf.toml exec python3 ../../../case_studies/test_program.py 2> alumet-agent-rapl+perf.log
```
Output: `alumet-output-rapl+perf.csv`
Logfile: `alumet-agent-rapl+perf.log`

## Process-specific Energy Measurement with Alumet

### Concept

In order to measure the per-process energy with Alumet, we need time series information of:

A. the total energy consumed by the hardware (CPU and/or GPU) - acquired through the `rapl` and `nvidia-nvml` source plugins

B. per-process usage of the hardware (CPU and/or GPU) - aqcuired through the `procfs` and/or `perf` source plugins

The energy consumed by the process in question is computed by a convolution of the two above two time series. This is achieved using the `energy-attribution` trasnform plugin. Higher measurement frequency of A. and B. should result in better accuracy of the computed process specific energy value. 

Lastly, the source and transformed values are written to a CSV file using `csv` plugin.

### Experiment with the `rapl`, `perf` and `energy-attribution` plugin

Folder: `experiments/03_rapl_perf_energy`

Command:
```bash
alumet-agent --config=alumet-config-rapl+perf+energy-attribution.toml exec python3 ../../../case_studies/test_program.py 2> alumet-agent-rapl+perf+energy-attribution.log
```
Output: `alumet-output-rapl+perf+energy-attribution.csv`
Logfile: `alumet-agent-rapl+perf+energy-attribution.log`


## Usage/Config of Alumet Plugins

- information on `CAP_SYS_NICE` from here https://alumet-dev.github.io/user-book/start/install.html#privileges-required

### 1. rapl
Special requirements for RAPL
- information on changing read access of `/sys/devices/virtual/powercap/intel-rapl` folder in filesystem from https://alumet-dev.github.io/user-book/plugins/1_sources/rapl.html?highlight=powercap#requirements
- it also uses `perf-events` under the hood for more accurate and high frequency measurements. 

### 2. nvidia-nvml
https://alumet-dev.github.io/user-book/plugins/1_sources/nvidia-nvml.html

### 3. procfs

Special requirements for procfs
- information on `hidepid` value and how to change it from here https://alumet-dev.github.io/user-book/plugins/1_sources/procfs.html#procfs-access

In order for `energy-attribution` to work, we need information on per process `cpu_percent` (% consumption wrt total consumption). To get this, we have to choose the `event` strategy in procfs processes. Below is an example:

```toml
[plugins.procfs.processes]
enabled = true
refresh_interval = "2s"
strategy = "event"
```

### 4. perf

Special requirements for perf
- information about `CAP_PERFMON` from https://alumet-dev.github.io/user-book/start/install.html#privileges-required
- information about `kernel.perf_event_paranoid` value from https://alumet-dev.github.io/user-book/plugins/1_sources/rapl.html#perf_event_paranoid-and-capabilities
- Note that this command will not make it permanent (reset after restart). To make it permanent, create a configuration file in /etc/sysctl.d/ (this may change depending on your Linux distro). Based on `README.sysctl` at `/etc/sysctl.d/`:
    >    My personal preference would be for local system settings to go into
    >    /etc/sysctl.d/local.conf but as long as you follow the rules for the names
    >    of the file, anything will work. See sysctl.conf(8) man page for details
    >    of the format.
    
    So we need to make a file `local.conf` with contents `kernel.perf_event_paranoid=2` at `/etc/sysctl.d/`


### 5. energy-attribution 

```
[plugins.energy-attribution.formulas.attributed_energy]
# the expression used to compute the final value
expr = "cpu_energy * cpu_usage / 100.0"
# the time reference: this is a timeseries, defined by a metric (and other criteria, see below), that will not change during the transformation. Other timeseries can be interpolated in order to have the same timestamps before applying the formula.
ref = "cpu_energy"

# Timeseries related to the resources.
[plugins.energy-attribution.formulas.attributed_energy.per_resource]
# Defines the timeseries `cpu_energy` that is used in the formula, as the measurement points that have:
# - the metric `rapl_consumed_energy`,
# - and the resource kind `"local_machine"`
# - and the attribute `domain` equal to `package_total`
cpu_energy = { metric = "rapl_consumed_energy", resource_kind = "local_machine", domain = "package_total" }

# Timeseries related to the resource consumers.
[plugins.energy-attribution.formulas.attributed_energy.per_consumer]
# Defines the timeseries `cpu_usage` that is used in the formula, as the measurements points that have:
# - the metric `cpu_percent`
# - the attribute `kind` equal to `total`
cpu_usage = { metric = "cpu_percent", kind = "total" }
```

### 6. csv

No details in the documentation. So configuration capabilities will need to be found from source code.


## Visualization Dashboard

To better visualize the measurement results, a thorough EDA is visualized through a visualization dashboard. 

Run
```bash
cd measurement_tools/alumet/viz
conda env create -f environment.yml # If using micromamba/mamba, replace conda with micromamba/mamba
python dashboard.py
```
