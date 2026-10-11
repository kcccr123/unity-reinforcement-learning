using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Kcccr.UnityReinforcementLearning
{
    /// <summary>
    /// Connects one Unity environment to the external Python trainer.
    /// It accepts episode resets and actions, then returns observations, rewards, and completion status.
    /// </summary>
    [AddComponentMenu("Reinforcement Learning/Training Host")]
    [DisallowMultipleComponent]
    public sealed class RlTrainingHost : MonoBehaviour
    {
        [SerializeField] private RlEnvironmentBehaviour environment = null;
        [SerializeField] private int port = 7777;
        [SerializeField] private bool autoStart = true;

        private readonly ConcurrentQueue<BridgeRequest> requests = new ConcurrentQueue<BridgeRequest>();
        private readonly ConcurrentQueue<string> workerErrors = new ConcurrentQueue<string>();
        private TcpListener listener;
        private volatile TcpClient client;
        private Thread worker;
        private BridgeRequest activeRequest;
        private volatile bool stopping;
        private volatile bool workerFinished;
        private bool episodeActive;
        private int observationSize;
        private int actionSize;

        /// <summary>
        /// Exposes the environment whose episodes this host presents to the trainer.
        /// </summary>
        public RlEnvironmentBehaviour Environment => environment;

        /// <summary>
        /// Exposes the port the Python trainer uses to connect to this host.
        /// </summary>
        public int Port => port;

        /// <summary>
        /// Reports whether this host is listening for or serving a trainer connection.
        /// </summary>
        public bool IsListening => listener != null && !workerFinished;

        /// <summary>
        /// Makes the assigned environment available when automatic startup is enabled at play start.
        /// </summary>
        private void Start()
        {
            if (autoStart)
            {
                StartTraining();
            }
        }

        /// <summary>
        /// Advances pending trainer requests through the Unity environment and returns completed results.
        /// </summary>
        private void Update()
        {
            while (workerErrors.TryDequeue(out string error))
            {
                Debug.LogError($"RL training connection: {error}", this);
            }

            if (workerFinished && listener != null)
            {
                StopTraining();
                return;
            }

            if (activeRequest != null)
            {
                CompleteActionIfReady();
                return;
            }

            if (requests.TryDequeue(out BridgeRequest request))
            {
                ProcessRequest(request);
            }
        }

        /// <summary>
        /// Ends the training session when this Unity component is disabled.
        /// </summary>
        private void OnDisable()
        {
            StopTraining();
        }

        /// <summary>
        /// Makes the assigned environment available to Python on the configured port.
        /// Returns true when the host is ready or already active, and false if it cannot start.
        /// </summary>
        public bool StartTraining()
        {
            if (listener != null)
            {
                if (!workerFinished)
                {
                    return true;
                }
                StopTraining();
            }

            if (environment == null || environment.ObservationSize <= 0 || environment.ActionSize <= 0)
            {
                Debug.LogError("Assign an RL environment with positive observation and action sizes.", this);
                return false;
            }

            if (port < 1 || port > 65535)
            {
                Debug.LogError("The RL training port must be between 1 and 65535.", this);
                return false;
            }

            observationSize = environment.ObservationSize;
            actionSize = environment.ActionSize;
            episodeActive = false;
            stopping = false;
            workerFinished = false;

            try
            {
                listener = new TcpListener(IPAddress.Loopback, port);
                listener.Start(1);
                TcpListener server = listener;
                worker = new Thread(() => RunSession(server)) { IsBackground = true, Name = "Unity RL TCP bridge" };
                worker.Start();
                Debug.Log($"RL training host listening on 127.0.0.1:{port}.", this);
                return true;
            }
            catch (Exception exception)
            {
                listener?.Stop();
                listener = null;
                Debug.LogError($"Could not start RL training host: {exception.Message}", this);
                return false;
            }
        }

        /// <summary>
        /// Ends the current training session so the environment no longer receives trainer requests.
        /// </summary>
        public void StopTraining()
        {
            stopping = true;
            listener?.Stop();
            client?.Close();
            activeRequest?.Reply.TrySetCanceled();
            activeRequest = null;

            while (requests.TryDequeue(out BridgeRequest request))
            {
                request.Reply.TrySetCanceled();
            }

            if (worker != null && worker.IsAlive)
            {
                worker.Join(1000);
            }

            worker = null;
            client = null;
            listener = null;
            episodeActive = false;
        }

        /// <summary>
        /// Uses the supplied listener to exchange requests and results with one Python trainer.
        /// </summary>
        private void RunSession(TcpListener server)
        {
            try
            {
                using (TcpClient accepted = server.AcceptTcpClient())
                {
                    client = accepted;
                    if (stopping)
                    {
                        return;
                    }
                    accepted.NoDelay = true;
                    using (NetworkStream stream = accepted.GetStream())
                    using (var reader = new StreamReader(stream, new UTF8Encoding(false)))
                    using (var writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true, NewLine = "\n" })
                    {
                        writer.WriteLine(RlProtocol.Handshake(observationSize, actionSize));

                        while (!stopping)
                        {
                            string line = reader.ReadLine();
                            if (line == null)
                            {
                                break;
                            }

                            var request = new BridgeRequest(line);
                            requests.Enqueue(request);
                            string response = request.Reply.Task.GetAwaiter().GetResult();
                            if (stopping)
                            {
                                break;
                            }

                            writer.WriteLine(response);
                            if (response.StartsWith("ERROR=", StringComparison.Ordinal))
                            {
                                break;
                            }
                        }
                    }
                }
            }
            catch (Exception exception)
            {
                if (!stopping)
                {
                    workerErrors.Enqueue(exception.Message);
                }
            }
            finally
            {
                workerFinished = true;
            }
        }

        /// <summary>
        /// Routes a trainer's reset or action request to the environment and prepares its response.
        /// </summary>
        private void ProcessRequest(BridgeRequest request)
        {
            if (RlProtocol.TryParseReset(request.Message, out int seed))
            {
                try
                {
                    environment.ResetEpisode(seed);
                    string response = RlProtocol.State(environment.CollectObservation(), observationSize, 0f, false);
                    episodeActive = true;
                    request.Reply.TrySetResult(response);
                }
                catch (Exception exception)
                {
                    FailRequest(request, exception);
                }
                return;
            }

            if (!episodeActive)
            {
                request.Reply.TrySetResult("ERROR=RESET_REQUIRED");
                return;
            }

            if (!RlProtocol.TryParseAction(request.Message, actionSize, out float[] action))
            {
                request.Reply.TrySetResult("ERROR=INVALID_ACTION");
                return;
            }

            try
            {
                environment.ApplyAction(action);
                activeRequest = request;
                CompleteActionIfReady();
            }
            catch (Exception exception)
            {
                FailRequest(request, exception);
            }
        }

        /// <summary>
        /// Returns the next observation, reward, and episode status once the environment's action finishes.
        /// </summary>
        private void CompleteActionIfReady()
        {
            BridgeRequest request = activeRequest;
            try
            {
                if (environment.IsActionRunning())
                {
                    return;
                }

                float reward = environment.CalculateReward(out bool isDone);
                string response = RlProtocol.State(environment.CollectObservation(), observationSize, reward, isDone);
                episodeActive = !isDone;
                activeRequest = null;
                request.Reply.TrySetResult(response);
            }
            catch (Exception exception)
            {
                FailRequest(request, exception);
            }
        }

        /// <summary>
        /// Reports an environment error to the trainer and ends the current episode.
        /// </summary>
        private void FailRequest(BridgeRequest request, Exception exception)
        {
            Debug.LogError($"RL environment failed: {exception.Message}", this);
            episodeActive = false;
            activeRequest = null;
            request.Reply.TrySetResult("ERROR=ENVIRONMENT");
        }

        /// <summary>
        /// Carries one trainer message and its eventual response between the connection and environment.
        /// </summary>
        private sealed class BridgeRequest
        {
            public readonly string Message;
            public readonly TaskCompletionSource<string> Reply =
                new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);

            /// <summary>
            /// Creates a request for the supplied trainer message, ready for an environment response.
            /// </summary>
            public BridgeRequest(string message)
            {
                Message = message;
            }
        }
    }
}
