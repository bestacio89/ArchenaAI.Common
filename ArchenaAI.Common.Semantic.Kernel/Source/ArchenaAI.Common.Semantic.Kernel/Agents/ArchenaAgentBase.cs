using System;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Messaging.Agents;
using ArchenaAI.Common.Semantic.Kernel.Orchestration;
using ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Agents
{
    /// <summary>
    /// Base class for ArchenaAI agents. Handles Kafka interaction, orchestration
    /// loops, response emission and correlation management.
    /// </summary>
    public abstract class ArchenaAgentBase : IAgent
    {
        private readonly ArchenaKernelOrchestrator _orchestrator;
        private readonly IKafkaConnector _kafka;
        private readonly ILogger _logger;

        protected ArchenaAgentBase(
            ArchenaKernelOrchestrator orchestrator,
            IKafkaConnector kafka,
            ILogger logger)
        {
            _orchestrator = orchestrator ?? throw new ArgumentNullException(nameof(orchestrator));
            _kafka = kafka ?? throw new ArgumentNullException(nameof(kafka));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        // -----------------------------------------------------------
        // Identification
        // -----------------------------------------------------------
        public abstract string Name { get; }

        /// <summary>
        /// Default topic for outbound agent messages.
        /// </summary>
        protected virtual string ResponseTopic => "archenaai.agents.responses";

        // -----------------------------------------------------------
        // Loop & Prompt Functions
        // -----------------------------------------------------------
        protected abstract LoopOptions GetLoopOptions();
        protected abstract string BuildPrompt(AgentEnvelope envelope);

        // -----------------------------------------------------------
        // Capability Matching
        // -----------------------------------------------------------
        /// <summary>
        /// Override to restrict which message types this agent can handle.
        /// By default, agents accept all message types.
        /// </summary>
        public virtual bool CanHandle(AgentMessageType type) => true;

        // -----------------------------------------------------------
        // Message Handling
        // -----------------------------------------------------------

     

        public async Task HandleAsync(AgentEnvelope envelope, CancellationToken ct)
        {
            if (envelope == null)
                throw new ArgumentNullException(nameof(envelope));

            _logger.LogInformation(
                "Agent '{Agent}' handling {CorrelationId} of type {Type}",
                Name, envelope.CorrelationId, envelope.Type);

            var prompt = BuildPrompt(envelope);
            var options = GetLoopOptions();

            string result;

            switch (envelope.Type)
            {
                case AgentMessageType.Request:
                case AgentMessageType.Think:
                    result = await _orchestrator.RunPlanExecuteLoopAsync(prompt, options, ct);
                    break;

                case AgentMessageType.Act:
                    result = await _orchestrator.RunReasoningLoopAsync(prompt, options, ct);
                    break;

                case AgentMessageType.MemoryIndex:
                case AgentMessageType.Normalize:
                case AgentMessageType.MemoryStore:
                case AgentMessageType.MemorySearch:
                    result = await _orchestrator.RunReasoningLoopAsync(prompt, options, ct);
                    break;

                default:
                    _logger.LogWarning("Agent '{Agent}' received unrecognized message type: {Type}",
                        Name, envelope.Type);
                    result = await _orchestrator.RunReasoningLoopAsync(prompt, options, ct);
                    break;
            }

            // -----------------------------------------------------------
            // Build Response Envelope
            // -----------------------------------------------------------
            var response = new AgentEnvelope
            {
                Type = AgentMessageType.Response,
                Agent = Name,
                CorrelationId = envelope.CorrelationId,
                ReplyTo = envelope.ReplyTo,
                Payload = result
            };

            response.Metadata["sourceAgent"] = Name;
            response.Metadata["requestType"] = envelope.Type.ToString();

            string json = AgentMessageSerializer.Serialize(response);
            string targetTopic = envelope.ReplyTo ?? ResponseTopic;

            _logger.LogInformation(
                "Agent '{Agent}' sending response for {CorrelationId} → {Topic}",
                Name, envelope.CorrelationId, targetTopic);

            await _kafka.PublishAsync(targetTopic, json, ct);
        }
    }
}
