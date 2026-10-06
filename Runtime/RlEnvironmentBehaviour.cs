using UnityEngine;

namespace Kcccr.UnityReinforcementLearning
{
    /// <summary>
    /// Use case: define one RL environment. Inputs: reset seed and action vector.
    /// Outputs: space sizes, observation, action status, reward, and done flag.
    /// </summary>
    public abstract class RlEnvironmentBehaviour : MonoBehaviour
    {
        public abstract int ObservationSize { get; }

        public abstract int ActionSize { get; }

        public abstract void ResetEpisode(int seed);

        public abstract float[] CollectObservation();

        public abstract void ApplyAction(float[] action);

        public abstract bool IsActionRunning();

        public abstract float CalculateReward(out bool isDone);
    }
}
