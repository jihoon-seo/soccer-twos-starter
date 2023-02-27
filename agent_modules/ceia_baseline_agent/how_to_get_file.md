# How to get `ceia_baseline_agent` module

You can get the pre-trained baseline agent module file at
- https://drive.google.com/uc?id=1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE <br>
  (Direct link; You can `wget` or `curl` with this URL.)
- https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view <br>
  (Google Drive UI; You can download or create a shortcut to your Google Drive.)

Extract the `ceia_baseline_agent.zip` file to this directory and run:

`python -m soccer_twos.watch -m1 agent_modules.onnx_agent -m2 agent_modules.ceia_baseline_agent`

The directory structure after unzipping should be like below:

```
soccer-twos-starter
├── README.md
├── agent_modules
│   ├── ceia_baseline_agent
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── agent_ray.py
│   │   ├── how_to_get_file.md
│   │   └── ray_results
│   │       └── PPO_selfplay_twos
│   │           ├── PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02
│   │           │   ├── checkpoint_002449
│   │           │   │   ├── checkpoint-2449
│   │           │   │   └── checkpoint-2449.tune_metadata
│   │           │   ├── events.out.tfevents.1632077642.jarbas
│   │           │   ├── params.json
│   │           │   ├── params.pkl
│   │           │   ├── progress.csv
│   │           │   └── result.json
│   │           ├── basic-variant-state-2021-09-19_15-54-01.json
│   │           └── experiment_state-2021-09-19_15-54-01.json
```
