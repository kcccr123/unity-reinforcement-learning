# unity-reinforcement-learning

Use case: Train one Unity agent through an external Python trainer. Implement `RlEnvironmentBehaviour` for your environment and assign it to `RlTrainingHost`; no game-specific environment is included.

Inputs: A reset seed, continuous action vector in `[-1, 1]`, environment callbacks, and training settings in `PythonEnv/example_config.yaml`.

Outputs: Observations, rewards, episode endings, checkpoints, a final model `.zip`, and an exported ONNX policy for local Unity inference.

Run: Install `PythonEnv/requirements.txt`, press Play in Unity, then run `python PythonEnv/train.py --config PythonEnv/example_config.yaml`. The counter sample is only for a smoke test.

Deploy: Install `PythonEnv/requirements-export.txt`, then run `python PythonEnv/export_onnx.py path/to/final_model.zip path/to/policy.onnx`. This exports a PPO MlpPolicy with deterministic, clipped actions, float32 `obs` and `actions` tensors, batch size 1, and ONNX opset 15. The checkpoint must use the bridge's continuous `[-1, 1]` action space and a flat observation vector.

Unity: The package declares Inference Engine 2.2.2 as its minimum dependency; the Unity 6000.3.2f1 game resolves 2.4.1. Import the ONNX file into the game's Assets, assign it to `RlModelRunner`, and call `Predict(float[] observation)` when the game needs an action. The runner keeps a CPU worker alive until its component is destroyed. Training uses `RlTrainingHost`; deployed inference only needs `RlModelRunner` and the game's observation/action mapping.

PCG examples: The sibling `kawaii-power-little-adventure` project contains `PcgRlEnvironment`, `Assets/Scenes/PcgTraining.unity`, and `Training/pcg.yaml` for room-count training. Its second example uses `PcgLayoutTraining.unity` and `Training/pcg-layout.yaml` for room count, width, and height. The corresponding `PcgModelDemo.unity` and `PcgLayoutModelDemo.unity` scenes run their saved ONNX models locally. See the game's `Training/README.md` and the README beside each demo for setup and results.
