using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace Kcccr.UnityReinforcementLearning
{
    /// <summary>Runs an exported single-observation policy locally, without the training host.</summary>
    [AddComponentMenu("Reinforcement Learning/Model Runner")]
    public sealed class RlModelRunner : MonoBehaviour
    {
        [SerializeField] private ModelAsset modelAsset;

        private Model model;
        private Worker worker;

        public float[] Predict(float[] observation)
        {
            if (model == null)
            {
                model = ModelLoader.Load(modelAsset);
            }

            // Reject a mismatched observation contract before creating or scheduling a worker.
            var inputShape = new TensorShape(1, observation.Length);
            DynamicTensorShape expectedShape = model.inputs[0].shape;
            if (expectedShape.IsStatic() && expectedShape.ToTensorShape() != inputShape)
            {
                throw new InvalidOperationException("Model expects input shape " + expectedShape
                    + ", but received " + inputShape + ". Train and assign a model matching the observation contract.");
            }

            // Reuse the CPU worker for subsequent predictions.
            if (worker == null)
            {
                worker = new Worker(model, BackendType.CPU);
            }

            using var input = new Tensor<float>(inputShape, observation);
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
