# Alumet for (Process Specific) Energy Measurement

Alumet (https://alumet.dev/) is a tool that allows measurement of data from various sources, transform them if necessary, and output them to various endpoints. 

## Installating Alumet

Here I proceed with installing Alumet from source. For this I needed to perform the following:

### Install Rust Tool Chain

The install instructions were available at https://rust-lang.org/tools/install/. 
On my Debian 13 linux machine, to I ran the command
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Everything went well!


### Install Alumet through `cargo`

The instructions were available at https://alumet-dev.github.io/user-book/start/install.html#option-4-installing-from-source

The specific command I used was: 
```
cargo install --git https://github.com/alumet-dev/alumet.git alumet-agent
```

This did not work on the first try, because the `pkg-config` and `OpenSSL` package were missing on my Debian 13 machine. I was able to fix it by installing them both with the commands

```
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

```
alumet-agent --plugins procfs,csv exec python3 -- -c "print(1+1)"
```

The dummy program we want to specifically look at is `python3 -c "print(1+1)"

Here by using `exec` command, the agent is first started, the command is executed, and when the command exits, the agent is stopped.

Following is the output from terminal:

```
ajc@mbd:~$ alumet-agent --plugins procfs,csv exec python3 -- -c "print(1+1)"
[2025-11-02T21:12:52Z INFO  alumet_agent] Starting Alumet agent 'alumet-agent' v0.9.1-e38f35d-dirty (2025-10-30T09:26:27.579386171Z, rustc 1.90.0, debug=false)
[2025-11-02T21:12:52Z INFO  alumet::agent::builder] Initializing the plugins...
[2025-11-02T21:12:52Z INFO  alumet::agent::builder] 2 plugins initialized.
[2025-11-02T21:12:52Z INFO  alumet::agent::builder] Starting the plugins...
[2025-11-02T21:12:52Z INFO  alumet::agent::builder] Plugin startup complete.
    🧩 2 plugins started:
        - csv v0.2.0
        - procfs v0.1.0
    
    ⭕ 24 plugins disabled:
        - aggregation v0.1.0
        - cgroups v0.1.0
        - elasticsearch v0.1.0
        - energy-attribution v0.1.0
        - energy-estimation-tdp v0.1.0
        - grace-hopper v0.1.0
        - influxdb v0.1.0
        - jetson v1.0.0
        - k8s v0.1.0
        - kwollect-input v1.0.0
        - kwollect-output v0.1.0
        - mongodb v0.1.0
        - nvml v0.4.0
        - oar v0.1.0
        - opentelemetry v0.1.0
        - perf v0.1.0
        - process-to-cgroup-bridge v0.1.0
        - prometheus-exporter v0.1.0
        - quarch v0.1.0
        - rapl v0.3.1
        - relay-client v0.6.0
        - relay-server v0.6.0
        - slurm v0.1.0
        - socket-control v0.2.1
    
    📏 16 metrics registered:
        - kernel_cpu_time: U64 (ms)
        - kernel_context_switches: U64 ()
        - kernel_new_forks: U64 ()
        - kernel_n_procs_running: U64 ()
        - kernel_n_procs_blocked: U64 ()
        - mem_total: U64 (kB)
        - mem_free: U64 (kB)
        - mem_available: U64 (kB)
        - cached: U64 (kB)
        - swap_cached: U64 (kB)
        - active: U64 (kB)
        - inactive: U64 (kB)
        - mapped: U64 (kB)
        - cpu_time_delta: U64 (ns)
        - cpu_percent: F64 ()
        - memory_usage: U64 (B)
    
    📥 2 sources, 🔀 0 transform and 📝 1 output registered.
    
    🔔 0 metric listener registered.
    
[2025-11-02T21:12:52Z INFO  alumet::agent::builder] Running pre-pipeline-start hooks...
[2025-11-02T21:12:52Z INFO  alumet::agent::builder] Starting the measurement pipeline...
[2025-11-02T21:12:52Z INFO  alumet::pipeline::builder] Only one output and no transform, using a simplified and optimized measurement pipeline.
[2025-11-02T21:12:52Z INFO  alumet::agent::builder] 🔥 ALUMET measurement pipeline has started.
[2025-11-02T21:12:52Z INFO  alumet::agent::builder] Running post-pipeline-start hooks...
[2025-11-02T21:12:52Z INFO  plugin_procfs] Starting system-wide process watcher.
[2025-11-02T21:12:52Z INFO  alumet::agent::builder] 🔥 ALUMET agent is ready.
[2025-11-02T21:12:52Z INFO  alumet::agent::exec] Child process 'python3' spawned with pid 408493.
2
[2025-11-02T21:12:52Z INFO  alumet::agent::exec] Child process exited with status exit status: 0, Alumet will now stop.
[2025-11-02T21:12:52Z INFO  alumet::agent::exec] Publishing EndConsumerMeasurement event
[2025-11-02T21:12:57Z INFO  alumet::agent::builder] Stopping the plugins...
[2025-11-02T21:12:57Z INFO  alumet::agent::builder] Stopping plugin csv v0.2.0
[2025-11-02T21:12:57Z INFO  alumet::agent::builder] Stopping plugin procfs v0.1.0
[2025-11-02T21:12:57Z INFO  alumet::agent::builder] All plugins have stopped.
```


Since we did not provide a `alumet-config.toml` for the plugins, the agent creates one for us, based on the plugins we specified. The agent also writes the output to `alumet-output.csv` file.

Contents of alumet-config.toml

+++ Unhide
```
[plugins.csv]
output_path = "alumet-output.csv"
force_flush = true
append_unit_to_metric_name = true
use_unit_display_name = true
csv_delimiter = ";"
csv_late_delimiter = ","

[plugins.procfs.kernel]
enabled = true
poll_interval = "5s"

[plugins.procfs.memory]
enabled = true
poll_interval = "5s"
metrics = [
    "MemTotal",
    "MemFree",
    "MemAvailable",
    "Cached",
    "SwapCached",
    "Active",
    "Inactive",
    "Mapped",
]

[plugins.procfs.processes]
enabled = true
refresh_interval = "2s"
strategy = "watcher"

[[plugins.procfs.processes.groups]]
exe_regex = ""
poll_interval = "2s"
flush_interval = "4s"

[plugins.procfs.processes.events]
poll_interval = "1s"
flush_interval = "4s"

```
+++

Contents of alumet-output.csv

+++ Unhide
```
metric;timestamp;value;resource_kind;resource_id;consumer_kind;consumer_id;__late_attributes
mem_total_kB;2025-11-02T21:12:52.714734289Z;16459247616;local_machine;;local_machine;;
mem_free_kB;2025-11-02T21:12:52.714734289Z;2673487872;local_machine;;local_machine;;
mem_available_kB;2025-11-02T21:12:52.714734289Z;6129909760;local_machine;;local_machine;;
cached_kB;2025-11-02T21:12:52.714734289Z;6440460288;local_machine;;local_machine;;
swap_cached_kB;2025-11-02T21:12:52.714734289Z;420552704;local_machine;;local_machine;;
active_kB;2025-11-02T21:12:52.714734289Z;7159353344;local_machine;;local_machine;;
inactive_kB;2025-11-02T21:12:52.714734289Z;3238080512;local_machine;;local_machine;;
mapped_kB;2025-11-02T21:12:52.714734289Z;1330573312;local_machine;;local_machine;;
mem_total_kB;2025-11-02T21:12:52.715023826Z;16459247616;local_machine;;local_machine;;
mem_free_kB;2025-11-02T21:12:52.715023826Z;2673487872;local_machine;;local_machine;;
mem_available_kB;2025-11-02T21:12:52.715023826Z;6129909760;local_machine;;local_machine;;
cached_kB;2025-11-02T21:12:52.715023826Z;6440460288;local_machine;;local_machine;;
swap_cached_kB;2025-11-02T21:12:52.715023826Z;420552704;local_machine;;local_machine;;
active_kB;2025-11-02T21:12:52.715023826Z;7159353344;local_machine;;local_machine;;
inactive_kB;2025-11-02T21:12:52.715023826Z;3238080512;local_machine;;local_machine;;
mapped_kB;2025-11-02T21:12:52.715023826Z;1330573312;local_machine;;local_machine;;
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;local_machine;;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;local_machine;;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;local_machine;;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;10;local_machine;;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;local_machine;;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;local_machine;;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;local_machine;;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;local_machine;;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;local_machine;;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;0;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;0;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;0;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;0;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;0;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;0;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;0;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;0;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;0;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;1;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;1;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;1;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;1;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;1;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;1;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;1;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;1;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;1;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;2;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;2;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;2;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;2;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;2;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;2;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;2;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;2;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;2;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;3;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;3;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;3;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;3;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;3;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;3;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;3;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;3;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;3;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;4;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;4;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;4;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;4;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;4;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;4;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;4;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;4;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;4;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;5;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;5;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;5;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;5;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;5;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;5;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;5;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;5;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;5;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;6;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;6;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;6;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;6;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;6;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;6;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;6;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;6;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;6;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;7;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;7;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;7;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;7;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;7;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;7;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;7;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;7;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;7;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;8;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;8;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;8;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;8;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;8;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;8;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;8;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;8;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;8;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;9;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;9;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;9;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;9;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;9;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;9;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;9;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;9;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;9;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;10;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;10;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;10;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;10;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;10;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;10;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;10;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;10;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;10;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;11;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;11;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;11;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;11;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;11;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;11;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;11;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;11;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.71509206Z;0;cpu_core;11;local_machine;;cpu_state=guest_nice
kernel_context_switches;2025-11-02T21:12:52.71509206Z;20;local_machine;;local_machine;;
kernel_new_forks;2025-11-02T21:12:52.71509206Z;2;local_machine;;local_machine;;
kernel_n_procs_running;2025-11-02T21:12:52.71509206Z;4;local_machine;;local_machine;;
kernel_n_procs_blocked;2025-11-02T21:12:52.71509206Z;1;local_machine;;local_machine;;
mem_total_kB;2025-11-02T21:12:52.761293384Z;16459247616;local_machine;;local_machine;;
mem_free_kB;2025-11-02T21:12:52.761293384Z;2665234432;local_machine;;local_machine;;
mem_available_kB;2025-11-02T21:12:52.761293384Z;6122987520;local_machine;;local_machine;;
cached_kB;2025-11-02T21:12:52.761293384Z;6434099200;local_machine;;local_machine;;
swap_cached_kB;2025-11-02T21:12:52.761293384Z;420552704;local_machine;;local_machine;;
active_kB;2025-11-02T21:12:52.761293384Z;7164678144;local_machine;;local_machine;;
inactive_kB;2025-11-02T21:12:52.761293384Z;3239411712;local_machine;;local_machine;;
mapped_kB;2025-11-02T21:12:52.761293384Z;1333768192;local_machine;;local_machine;;
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;80;local_machine;;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;20;local_machine;;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;70;local_machine;;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;360;local_machine;;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;local_machine;;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;local_machine;;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;local_machine;;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;local_machine;;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;local_machine;;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;0;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;10;cpu_core;0;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;40;cpu_core;0;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;0;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;0;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;0;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;0;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;0;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;0;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;1;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;1;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;1;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;40;cpu_core;1;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;1;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;1;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;1;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;1;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;1;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;30;cpu_core;2;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;2;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;2;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;20;cpu_core;2;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;2;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;2;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;2;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;2;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;2;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;3;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;3;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;20;cpu_core;3;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;30;cpu_core;3;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;3;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;3;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;3;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;3;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;3;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;4;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;4;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;4;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;40;cpu_core;4;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;4;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;4;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;4;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;4;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;4;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;5;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;5;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;10;cpu_core;5;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;40;cpu_core;5;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;5;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;5;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;5;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;5;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;5;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;6;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;6;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;6;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;40;cpu_core;6;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;6;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;6;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;6;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;6;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;6;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;20;cpu_core;7;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;7;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;7;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;30;cpu_core;7;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;7;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;7;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;7;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;7;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;7;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;30;cpu_core;8;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;8;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;20;cpu_core;8;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;8;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;8;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;8;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;8;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;8;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;8;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;9;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;10;cpu_core;9;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;9;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;30;cpu_core;9;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;9;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;9;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;9;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;9;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;9;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;10;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;10;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;10;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;50;cpu_core;10;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;10;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;10;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;10;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;10;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;10;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;11;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;11;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;11;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;50;cpu_core;11;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;11;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;11;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;11;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;11;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:52.761398959Z;0;cpu_core;11;local_machine;;cpu_state=guest_nice
kernel_context_switches;2025-11-02T21:12:52.761398959Z;445;local_machine;;local_machine;;
kernel_new_forks;2025-11-02T21:12:52.761398959Z;4;local_machine;;local_machine;;
kernel_n_procs_running;2025-11-02T21:12:52.761398959Z;4;local_machine;;local_machine;;
kernel_n_procs_blocked;2025-11-02T21:12:52.761398959Z;0;local_machine;;local_machine;;
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;460;local_machine;;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;80;local_machine;;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;330;local_machine;;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;58140;local_machine;;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;local_machine;;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;local_machine;;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;local_machine;;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;local_machine;;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;local_machine;;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;80;cpu_core;0;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;10;cpu_core;0;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;60;cpu_core;0;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;4750;cpu_core;0;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;0;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;0;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;0;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;0;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;0;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;1;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;1;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;10;cpu_core;1;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;4890;cpu_core;1;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;1;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;1;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;1;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;1;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;1;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;90;cpu_core;2;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;2;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;50;cpu_core;2;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;4750;cpu_core;2;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;2;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;2;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;2;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;2;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;2;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;10;cpu_core;3;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;3;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;3;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;4930;cpu_core;3;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;3;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;3;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;3;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;3;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;3;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;80;cpu_core;4;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;4;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;20;cpu_core;4;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;4810;cpu_core;4;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;4;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;4;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;4;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;4;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;4;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;20;cpu_core;5;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;5;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;40;cpu_core;5;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;4860;cpu_core;5;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;5;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;5;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;5;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;5;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;5;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;40;cpu_core;6;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;10;cpu_core;6;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;20;cpu_core;6;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;4870;cpu_core;6;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;6;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;6;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;6;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;6;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;6;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;40;cpu_core;7;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;7;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;30;cpu_core;7;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;4850;cpu_core;7;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;7;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;7;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;7;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;7;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;7;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;10;cpu_core;8;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;30;cpu_core;8;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;20;cpu_core;8;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;4830;cpu_core;8;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;8;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;8;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;8;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;8;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;8;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;20;cpu_core;9;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;9;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;30;cpu_core;9;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;4880;cpu_core;9;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;9;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;9;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;9;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;9;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;9;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;60;cpu_core;10;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;40;cpu_core;10;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;20;cpu_core;10;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;4840;cpu_core;10;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;10;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;10;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;10;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;10;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;10;local_machine;;cpu_state=guest_nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;20;cpu_core;11;local_machine;;cpu_state=user
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;11;local_machine;;cpu_state=nice
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;10;cpu_core;11;local_machine;;cpu_state=system
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;4900;cpu_core;11;local_machine;;cpu_state=idle
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;11;local_machine;;cpu_state=irq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;11;local_machine;;cpu_state=softirq
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;11;local_machine;;cpu_state=steal
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;11;local_machine;;cpu_state=guest
kernel_cpu_time_ms;2025-11-02T21:12:57.715114602Z;0;cpu_core;11;local_machine;;cpu_state=guest_nice
kernel_context_switches;2025-11-02T21:12:57.715114602Z;8997;local_machine;;local_machine;;
kernel_new_forks;2025-11-02T21:12:57.715114602Z;6;local_machine;;local_machine;;
kernel_n_procs_running;2025-11-02T21:12:57.715114602Z;2;local_machine;;local_machine;;
kernel_n_procs_blocked;2025-11-02T21:12:57.715114602Z;0;local_machine;;local_machine;;
mem_total_kB;2025-11-02T21:12:57.715265335Z;16459247616;local_machine;;local_machine;;
mem_free_kB;2025-11-02T21:12:57.715265335Z;2681049088;local_machine;;local_machine;;
mem_available_kB;2025-11-02T21:12:57.715265335Z;6139310080;local_machine;;local_machine;;
cached_kB;2025-11-02T21:12:57.715265335Z;6417108992;local_machine;;local_machine;;
swap_cached_kB;2025-11-02T21:12:57.715265335Z;420589568;local_machine;;local_machine;;
active_kB;2025-11-02T21:12:57.715265335Z;7173455872;local_machine;;local_machine;;
inactive_kB;2025-11-02T21:12:57.715265335Z;3239809024;local_machine;;local_machine;;
mapped_kB;2025-11-02T21:12:57.715265335Z;1340239872;local_machine;;local_machine;;
```
+++

### Experiments

#### Fiddling with the `rapl` plugin
Command: 
```
alumet-agent --config=alumet-config-rapl.toml exec python3 program.py 2> alumet-agent-rapl.log
```
Output: `alumet-output-rapl.csv`

#### Fiddling with the `rapl` and `perf` plugin
Command: 
```
alumet-agent --config=alumet-config-rapl+perf.toml exec python3 program.py 2> alumet-agent-rapl+perf.log
```
Output: `alumet-output-rapl+perf.csv`

## Process Specific Energy Measurement with Alumet

In order to measure the per-process energy with Alumet, we need time series information of:

A. the total energy consumed by the hardware (CPU and/or GPU) - acquired through the `rapl` and `nvidia-nvml` source plugins
B. per-process usage of the hardware (CPU and/or GPU) - aqcuired through the `procfs` and/or `perf` source plugins

The energy consumed by the process in question is a convolution of the two above two time series. This is achieved using the `energy-attribution` trasnform plugin. Higher measurement frequency of A. and B. should result in better accuracy of the computed process specific energy value. 

Lastly, the source and transformed values are written to a CSV file using `csv` plugin.

### Experiment with the `rapl`, `perf` and `energy-attribution` plugin
Command:
```
alumet-agent --config=alumet-config-rapl+perf+energy-attribution.toml exec python3 program.py 2> alumet-agent-rapl+perf+energy-attribution.log
```
Output: `alumet-output-rapl+perf+energy-attribution.csv`


## Usage/Config of Alumet Plugins

- information on `CAP_SYS_NICE` from here https://alumet-dev.github.io/user-book/start/install.html#privileges-required

### rapl
Special requirements for RAPL
- information on changing read access of `/sys/devices/virtual/powercap/intel-rapl` folder in filesystem from https://alumet-dev.github.io/user-book/plugins/1_sources/rapl.html?highlight=powercap#requirements
- it also uses `perf-events` under the hood for more accurate and high frequency measurements. 

### nvidia-nvml
https://alumet-dev.github.io/user-book/plugins/1_sources/nvidia-nvml.html

### procfs

Special requirements for procfs
- information on `hidepid` value and how to change it from here https://alumet-dev.github.io/user-book/plugins/1_sources/procfs.html#procfs-access

In order for `energy-attribution` to work, we need information on per process `cpu_percent` (% consumption wrt total consumption). To get this, we have to choose the `event` strategy in procfs processes. Below is an example:

```
[plugins.procfs.processes]
enabled = true
refresh_interval = "2s"
strategy = "event"
```

### perf

Special requirements for perf
- information about `CAP_PERFMON` from https://alumet-dev.github.io/user-book/start/install.html#privileges-required
- information about `perf_event_paranoid` value from https://alumet-dev.github.io/user-book/plugins/1_sources/rapl.html#perf_event_paranoid-and-capabilities
- Note that this command will not make it permanent (reset after restart). To make it permanent, create a configuration file in /etc/sysctl.d/ (this may change depending on your Linux distro).


### energy-attribution 

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

### csv

No details in the documentation. So configuration capabilities will need to be found from source code.