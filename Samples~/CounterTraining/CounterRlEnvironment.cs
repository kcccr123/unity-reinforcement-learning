using Kcccr.UnityReinforcementLearning;

/// <summary>
/// Use case: smoke-test single-agent training. Inputs: reset seed and one action value.
/// Outputs: one observation, reward, and episode ending.
/// </summary>
public sealed class CounterRlEnvironment : RlEnvironmentBehaviour
{
    private int steps;
    private float lastReward;

    public override int ObservationSize => 1;

    public override int ActionSize => 1;

    public override void ResetEpisode(int seed)
    {
        steps = 0;
        lastReward = 0f;
    }

    public override float[] CollectObservation()
    {
        return new[] { steps / 2f };
    }

    public override void ApplyAction(float[] action)
    {
        steps++;
        lastReward = action[0] >= 0f ? 1f : 0f;
    }

    public override bool IsActionRunning()
    {
        return false;
    }

    public override float CalculateReward(out bool isDone)
    {
        isDone = steps >= 2;
        return lastReward;
    }
}
