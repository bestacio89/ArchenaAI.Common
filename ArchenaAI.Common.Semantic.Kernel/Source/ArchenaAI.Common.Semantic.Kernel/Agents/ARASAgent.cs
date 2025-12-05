using ArchenaAI.Common.Semantic.Kernel.Messaging.Agents;
using ArchenaAI.Common.Semantic.Kernel.Orchestration;
using ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Agents
{
    public sealed class ARASAgent : ArchenaAgentBase
    {
        public ARASAgent(
            ArchenaKernelOrchestrator orchestrator,
            IKafkaConnector kafka,
            ILogger<ARASAgent> logger)
            : base(orchestrator, kafka, logger)
        {
        }

        public override string Name => "aras";

        public override bool CanHandle(AgentMessageType type) =>
            type is AgentMessageType.ComplianceCheck
                 or AgentMessageType.PolicyEvaluation
                 or AgentMessageType.Request
                 or AgentMessageType.Think;

        protected override LoopOptions GetLoopOptions() =>
            new()
            {
                MaxIterations = 3,
                UseMemory = false,
                UseCorrection = true,
                UseReflection = true,
            };

        protected override string BuildPrompt(AgentEnvelope envelope)
        {
            return $@"
You are the ARAS Agent (Architecture Rules And Safety).

Your job:
 - Enforce system constraints
 - Validate architecture decisions
 - Detect unsafe or non-compliant patterns
 - Ensure domain rules are respected

Payload:
{envelope.Payload}
";
        }
    }
}
