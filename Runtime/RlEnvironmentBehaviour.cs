using UnityEngine;

namespace Kcccr.UnityReinforcementLearning
{
    /// <summary>
    /// Defines the game-specific callbacks the training host uses for each single-agent episode.
    /// Implementations receive reset seeds and actions, then provide observations, rewards, and completion status.
    /// </summary>
    public abstract class RlEnvironmentBehaviour : MonoBehaviour
    {
        /// <summary>
        /// Tells the host how many values each observation sent to the trainer must contain.
        /// </summary>
        public abstract int ObservationSize { get; }

        /// <summary>
        /// Tells the host how many action values to expect from the trainer at each step.
        /// </summary>
        public abstract int ActionSize { get; }

        /// <summary>
        /// Prepares a new episode from the supplied seed before the host requests its first observation.
        /// </summary>
        public abstract void ResetEpisode(int seed);

        /// <summary>
        /// Returns the current state to the trainer after a reset or completed action.
        /// The observation must contain <see cref="ObservationSize"/> values.
        /// </summary>
        public abstract float[] CollectObservation();

        /// <summary>
        /// Applies the trainer's next action to the environment.
        /// The action must contain <see cref="ActionSize"/> values in [-1, 1].
        /// </summary>
        public abstract void ApplyAction(float[] action);

        /// <summary>
        /// Lets the host wait for an action to finish before collecting its result.
        /// Returns true while the action is still running.
        /// </summary>
        public abstract bool IsActionRunning();

        /// <summary>
        /// Gives the host the completed action's reward and whether the episode has ended.
        /// Both results are returned to the trainer with the next observation.
        /// </summary>
        public abstract float CalculateReward(out bool isDone);
    }
}
