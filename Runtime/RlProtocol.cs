using System;
using System.Globalization;

namespace Kcccr.UnityReinforcementLearning
{
    /// <summary>
    /// Defines the messages exchanged between the Unity host and Python trainer for one agent.
    /// It carries environment dimensions, reset and action requests, and step results.
    /// </summary>
    internal static class RlProtocol
    {
        private const string ResetPrefix = "ACT=RESET;SEED=";
        private const string ActionPrefix = "ACT=";

        /// <summary>
        /// Sends the observation and action sizes Python needs to set up its training environment.
        /// </summary>
        public static string Handshake(int observationSize, int actionSize)
        {
            return $"CONFIG:OBS={observationSize};ACT={actionSize};ENV_TYPE=SINGLE";
        }

        /// <summary>
        /// Reads the seed from a trainer request so the host can begin a new episode.
        /// Returns false if the message is not a valid reset request.
        /// </summary>
        public static bool TryParseReset(string message, out int seed)
        {
            seed = 0;
            return message.StartsWith(ResetPrefix, StringComparison.Ordinal) &&
                int.TryParse(message.Substring(ResetPrefix.Length), NumberStyles.Integer,
                    CultureInfo.InvariantCulture, out seed);
        }

        /// <summary>
        /// Reads the trainer's next action for the host to apply to the environment.
        /// Returns false unless it contains the expected number of values in [-1, 1].
        /// </summary>
        public static bool TryParseAction(string message, int actionSize, out float[] action)
        {
            action = null;
            if (!message.StartsWith(ActionPrefix, StringComparison.Ordinal))
            {
                return false;
            }

            string[] parts = message.Substring(ActionPrefix.Length).Split(',');
            if (parts.Length != actionSize)
            {
                return false;
            }

            var values = new float[actionSize];
            for (int index = 0; index < actionSize; index++)
            {
                if (!float.TryParse(parts[index], NumberStyles.Float, CultureInfo.InvariantCulture,
                    out float value) || float.IsNaN(value) || float.IsInfinity(value) ||
                    value < -1f || value > 1f)
                {
                    return false;
                }
                values[index] = value;
            }

            action = values;
            return true;
        }

        /// <summary>
        /// Returns an observation, reward, and episode-end flag to the trainer after a reset or step.
        /// The observation must contain the expected number of values.
        /// </summary>
        public static string State(float[] observation, int observationSize, float reward, bool isDone)
        {
            if (observation == null || observation.Length != observationSize ||
                float.IsNaN(reward) || float.IsInfinity(reward))
            {
                throw new ArgumentException("Invalid RL observation or reward.");
            }

            var values = new string[observationSize];
            for (int index = 0; index < observationSize; index++)
            {
                float value = observation[index];
                if (float.IsNaN(value) || float.IsInfinity(value))
                {
                    throw new ArgumentException("RL observations must contain finite values.");
                }
                values[index] = value.ToString("R", CultureInfo.InvariantCulture);
            }

            return $"OBS={string.Join(",", values)};REW={reward.ToString("R", CultureInfo.InvariantCulture)};DONE={(isDone ? 1 : 0)}";
        }
    }
}
