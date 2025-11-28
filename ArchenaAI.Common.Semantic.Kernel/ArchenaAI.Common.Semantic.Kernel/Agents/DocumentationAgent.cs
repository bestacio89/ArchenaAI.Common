using ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka;
using ArchenaAI.Common.Semantic.Kernel.Messaging.Agents;
using ArchenaAI.Common.Semantic.Kernel.Orchestration;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Agents
{
    /// <summary>
    /// Agent specialized in producing high-quality technical documentation,
    /// architecture decision records, and narrative explanations.
    /// </summary>
    public sealed class DocumentationAgent : ArchenaAgentBase
    {
        public DocumentationAgent(
            ArchenaKernelOrchestrator orchestrator,
            IKafkaConnector kafka,
            ILogger<DocumentationAgent> logger)
            : base(orchestrator, kafka, logger)
        {
        }

        public override string Name => "documentation";

        protected override LoopOptions GetLoopOptions()
        {
            return new LoopOptions
            {
                MaxIterations = 3,
                UseMemory = true,
                UseReflection = true,
                UseCorrection = true,
                TerminationRegex = @"\b(end of document|end of doc|final documentation)\b"
            };
        }
        public override bool CanHandle(AgentMessageType type) =>
    type is AgentMessageType.Request
         or AgentMessageType.Think
         or AgentMessageType.Act
         or AgentMessageType.Normalize
         or AgentMessageType.Documentation;

        protected override string BuildPrompt(AgentEnvelope envelope)
        {
            return
$@"You are the ArchenaAI Documentation Agent.

You produce:
- Clear, structured technical documentation
- Architecture decision records (ADR)
- High-level and low-level design descriptions
- API docs, sequence explanations, and operational guides

User task:
{envelope.Payload}";
        }
    }
}
