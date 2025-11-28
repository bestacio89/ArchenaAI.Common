using ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka;
using ArchenaAI.Common.Semantic.Kernel.Messaging.Agents;
using ArchenaAI.Common.Semantic.Kernel.Orchestration;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Agents
{
    /// <summary>
    /// Agent specialized in CI/CD, IaC, cloud pipelines, and runtime operations.
    /// </summary>
    public sealed class DevOpsAgent : ArchenaAgentBase
    {
        public DevOpsAgent(
            ArchenaKernelOrchestrator orchestrator,
            IKafkaConnector kafka,
            ILogger<DevOpsAgent> logger)
            : base(orchestrator, kafka, logger)
        {
        }

        public override string Name => "devops";

        protected override LoopOptions GetLoopOptions()
        {
            return new LoopOptions
            {
                MaxIterations = 3,
                UseMemory = true,
                UseReflection = true,
                UseCorrection = true,
                TerminationRegex = @"\b(final pipeline|final yaml|deployment plan complete)\b"
            };
        }

        public override bool CanHandle(AgentMessageType type) =>
         type is AgentMessageType.Act
         or AgentMessageType.DevOpsTask
         or AgentMessageType.Request;


        protected override string BuildPrompt(AgentEnvelope envelope)
        {
            return
$@"You are the ArchenaAI DevOps Agent.

You design and reason about:
- CI/CD pipelines (Azure DevOps, GitHub, GitLab, etc.)
- IaC (Bicep, ARM, Terraform, etc.)
- Observability, deployment strategies, and rollbacks
- Alignment with Franz-based microservices and environments

User task:
{envelope.Payload}";
        }
    }
}
