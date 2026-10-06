using System;
using System.Globalization;

namespace Kcccr.UnityReinforcementLearning
{
    internal static class RlProtocol
    {
        private const string ResetPrefix = "ACT=RESET;SEED=";
        private const string ActionPrefix = "ACT=";

        public static string Handshake(int observationSize, int actionSize)
        {
            return $"CONFIG:OBS={observationSize};ACT={actionSize};ENV_TYPE=SINGLE";
        }

        public static bool TryParseReset(string message, out int seed)
        {
            seed = 0;
            return message.StartsWith(ResetPrefix, StringComparison.Ordinal) &&
                int.TryParse(message.Substring(ResetPrefix.Length), NumberStyles.Integer,
                    CultureInfo.InvariantCulture, out seed);
        }

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
