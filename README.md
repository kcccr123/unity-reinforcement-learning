# Unity reinforcement learning

Train one agent in a developer-defined Unity environment through an external Python trainer, then run supported exported policies locally. The framework supplies the bridge, trainer, exporter, and model runner. The developer supplies the task and observation/action meanings.

## Initial package setup

1. Install `package.json` through Unity Package Manager's **Install package from disk** command. Code in an assembly definition references `Kcccr.UnityReinforcementLearning.Runtime`.
2. Prepare a Python environment and install `PythonEnv/requirements.txt` from this repository. Install `PythonEnv/requirements-export.txt` when exporting ONNX.
3. Create a training-config/artifact directory in the project that owns the model. Copy `PythonEnv/example_config.yaml` there as a template.

## Implement an environment

Derive a component from `RlEnvironmentBehaviour` and implement these callbacks:

| Contract | Developer responsibility |
| --- | --- |
| `ObservationSize`, `ActionSize` | Fixed vector lengths. Observations are flat finite floats; continuous actions are in `[-1, 1]`. |
| `ResetEpisode(seed)` | Prepare a new episode and reset tracking. |
| `CollectObservation()` | Encode current state in a documented field order. |
| `ApplyAction(action)` | Decode and apply one action. |
| `IsActionRunning()` | Keep returning true until the action has finished. |
| `CalculateReward(out isDone)` | Supply the learning reward and episode-ending flag. |

Add **Reinforcement Learning > Training Host** to a scene, assign the environment, and choose a port. Match it in the model-owned YAML. The host defaults to 7777 and dispatches Unity callbacks on the main thread. It calculates reward/done before collecting the post-action observation; preserve the terminal snapshot until returned. The included counter sample is available to check the bridge.

## Run training

```text
+------------------------------------------------------------------+
| PYTHON: trainer + UnityGymEnv                                    |
| Load model-owned YAML.                                           |
| Save model-owned checkpoints and final_model.zip.                |
+------------------------------------------------------------------+
             | reset / action
             ^ observation / reward / done
             | loopback TCP
             v
+------------------------------------------------------------------+
| UNITY: RlTrainingHost                                            |
| Dispatch callbacks to the developer's Unity environment.         |
+------------------------------------------------------------------+
             |
             v
+------------------------------------------------------------------+
| DEVELOPER ENVIRONMENT                                            |
| Reset -> observe -> apply action -> reward / done.               |
+------------------------------------------------------------------+
```

Start the Unity scene, then run:

```shell
python path/to/unity-reinforcement-learning/PythonEnv/train.py --config path/to/model-project/Training/agent.yaml
```

The YAML selects PPO/A2C, seed, learning parameters, timestep budget, device, checkpoint frequency, output directory, and optional compatible checkpoint resumption. Optional `ent_coef` controls entropy regularization; it defaults to zero for fresh policies. If specified while resuming, it overrides the saved coefficient. Relative `output_dir` and `resume_from` paths resolve beside that YAML; absolute paths are accepted. Each run writes a timestamped directory. Keep project configs, checkpoints, and results with their model owner rather than in this library.

The trainer uses feedforward `MlpPolicy`. A timestep budget alone lets PPO complete its final rollout. Set the optional `max_episodes` to end at an exact completed-episode count, with a `total_timesteps` budget large enough to reach it. The trainer stops before starting another Unity episode and saves the last completed PPO update. Restart Play/the player or restart the host before another trainer connection.

Set the optional `metrics_dir` YAML path to also write Stable-Baselines3 logging snapshots to `progress.json`, one JSON object per line. Console progress remains available. Relative paths resolve beside the config. Give each invocation a separate metrics directory to retain its metrics history.

## Export and deploy

```shell
python path/to/unity-reinforcement-learning/PythonEnv/export_onnx.py path/to/final_model.zip path/to/model-project/Assets/Models/agent.onnx
```

The exporter supports PPO `MlpPolicy` with flat observations and continuous `[-1, 1]` actions. It exports deterministic clipped actions with float32 `obs`/`actions`, batch size 1, and ONNX opset 15.

Import the asset, add **Reinforcement Learning > Model Runner**, and assign it. A developer-written policy adapter calls `Predict(float[] observation)` and applies the result using the same encoding/decoding as training. The runner rejects incompatible static input shapes and reuses a CPU worker until destroyed. Local inference needs neither Python nor a training host. Compare exported actions with the checkpoint, then evaluate task performance on separate scenarios.

## Examples

The sibling game's [training guide](../kawaii-power-little-adventure/Training/README.md) and [integration design](../kawaii-power-little-adventure/Training/PCG-system-design.md) demonstrate actor control and generation-parameter selection. Their sensors, rewards, scenes, and model contracts are game-specific examples. Optional final-evidence reporting uses the separate [QA framework](../rein-powered-QA-framework/README.md).
