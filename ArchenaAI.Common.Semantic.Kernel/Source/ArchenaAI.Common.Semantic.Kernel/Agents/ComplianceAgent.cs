using ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka;
using ArchenaAI.Common.Semantic.Kernel.Messaging.Agents;
using ArchenaAI.Common.Semantic.Kernel.Orchestration;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Agents
{
    /// <summary>
    /// Agent focused on compliance, governance, and Aegis/Franz alignment.
    /// Intended to validate architectures, pipelines, and implementations
    /// against predefined rules and regulatory constraints.
    /// </summary>
    public sealed class ComplianceAgent : ArchenaAgentBase
    {
        public ComplianceAgent(
            ArchenaKernelOrchestrator orchestrator,
            IKafkaConnector kafka,
            ILogger<ComplianceAgent> logger)
            : base(orchestrator, kafka, logger)
        {
        }

        public override string Name => "compliance";

        protected override LoopOptions GetLoopOptions()
        {
            return new LoopOptions
            {
                MaxIterations = 4,
                UseMemory = true,
                UseReflection = true,
                UseCorrection = true,
                TerminationRegex = @"\b(compliance verdict|final assessment|go/no-go decision)\b"
            };
        }

        protected override string BuildPrompt(AgentEnvelope envelope)
        {
            return
$@"You are the ArchenaAI Compliance and Governance Agent.

You evaluate:
- Alignment with architecture rules (Franz/Aegis)
- Regulatory and organizational constraints
- Security, privacy, and operational risk implications
- Whether a proposal is acceptable, needs revision, or should be rejected

User task:
{envelope.Payload}";
        }
    }
}
