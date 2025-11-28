using ArchenaAI.Common.Semantic.Kernel.Messaging.Agents;
using ArchenaAI.Common.Semantic.Kernel.Orchestration;
using ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Agents
{
    public sealed class SREAgent : ArchenaAgentBase
    {
        public SREAgent(
            ArchenaKernelOrchestrator orchestrator,
            IKafkaConnector kafka,
            ILogger<SREAgent> logger)
            : base(orchestrator, kafka, logger)
        {
        }

        public override string Name => "sre";

        public override bool CanHandle(AgentMessageType type) =>
            type is AgentMessageType.Incident
                 or AgentMessageType.LogAnalysis
                 or AgentMessageType.Request
                 or AgentMessageType.Think
                 or AgentMessageType.Act;

        protected override LoopOptions GetLoopOptions() =>
            new()
            {
                MaxIterations = 4,
                UseMemory = true,
                UseCorrection = true,
                UseReflection = true
            };

        protected override string BuildPrompt(AgentEnvelope envelope)
        {
            return $@"
You are the SRE Agent in the ArchenaAI ecosystem.

Your duties:
 - Diagnose incidents
 - Interpret logs
 - Detect reliability issues
 - Recommend recovery actions
 - Improve resilience

User payload:
{envelope.Payload}
";
        }
    }
}
