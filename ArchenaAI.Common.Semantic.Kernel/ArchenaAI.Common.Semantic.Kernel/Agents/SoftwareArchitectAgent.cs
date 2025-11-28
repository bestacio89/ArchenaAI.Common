using ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka;
using ArchenaAI.Common.Semantic.Kernel.Messaging.Agents;
using ArchenaAI.Common.Semantic.Kernel.Orchestration;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Agents
{
    /// <summary>
    /// Agent specialized in software and solution architecture:
    /// microservices, DDD, event-driven systems, Franz conventions, etc.
    /// </summary>
    public sealed class SoftwareArchitectAgent : ArchenaAgentBase
    {
        public SoftwareArchitectAgent(
            ArchenaKernelOrchestrator orchestrator,
            IKafkaConnector kafka,
            ILogger<SoftwareArchitectAgent> logger)
            : base(orchestrator, kafka, logger)
        {
        }

        public override string Name => "software-architect";

        public override bool CanHandle(AgentMessageType type) =>
        type is AgentMessageType.Request
        or AgentMessageType.Think
        or AgentMessageType.Act;
        protected override LoopOptions GetLoopOptions()
        {
            return new LoopOptions
            {
                MaxIterations = 4,
                UseMemory = true,
                UseReflection = true,
                UseCorrection = true,
                TerminationRegex = @"\b(final architecture|final design|end of proposal)\b"
            };
        }

        protected override string BuildPrompt(AgentEnvelope envelope)
        {
            // Here you inject your "Bernardo-level" persona
            return
$@"You are the ArchenaAI Software Architect Agent.

You design event-driven, microservice-based, Franz-compliant architectures with:
- .NET 10 / Franz.Common building blocks
- Clean Architecture, DDD, CQRS, resilience and observability
- Kafka-first integration, multi-tenant SaaS and deterministic governance

User task:
{envelope.Payload}";
        }
    }
}
