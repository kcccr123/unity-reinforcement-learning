using Unity.InferenceEngine;
using UnityEngine;

namespace Kcccr.UnityReinforcementLearning
{
    /// <summary>Runs an exported single-observation policy locally, without the training host.</summary>
    [AddComponentMenu("Reinforcement Learning/Model Runner")]
    public sealed class RlModelRunner : MonoBehaviour
    {
        [SerializeField] private ModelAsset modelAsset;

        private Worker worker;

        public float[] Predict(float[] observation)
        {
            // Load once; each dungeon requests just one inference on the CPU.
            if (worker == null)
            {
                worker = new Worker(ModelLoader.Load(modelAsset), BackendType.CPU);
            }

            using var input = new Tensor<float>(new TensorShape(1, observation.Length), observation);
            worker.Schedule(input);
            var output = (Tensor<float>)worker.PeekOutput();
            // The worker owns the output tensor; return an independent copy of its values.
            return output.DownloadToArray();
        }

        private void OnDestroy()
        {
            worker?.Dispose();
        }
    }
}
