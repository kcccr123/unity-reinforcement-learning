# unity-reinforcement-learning

Use case: Train one Unity agent through an external Python trainer. Implement `RlEnvironmentBehaviour` for your environment and assign it to `RlTrainingHost`; no game-specific environment is included.

Inputs: A reset seed, continuous action vector in `[-1, 1]`, environment callbacks, and training settings in `PythonEnv/example_config.yaml`.

Outputs: Observations, rewards, episode endings, checkpoints, and a final model `.zip`.

Run: Install `PythonEnv/requirements.txt`, press Play in Unity, then run `python PythonEnv/train.py --config PythonEnv/example_config.yaml`. The counter sample is only for a smoke test.
