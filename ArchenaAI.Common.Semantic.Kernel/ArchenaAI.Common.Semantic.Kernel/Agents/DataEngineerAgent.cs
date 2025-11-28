using ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka;
using ArchenaAI.Common.Semantic.Kernel.Messaging.Agents;
using ArchenaAI.Common.Semantic.Kernel.Orchestration;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Agents
{
    /// <summary>
    /// Agent specialized in data flows, ETL/ELT, warehousing, and analytics pipelines.
    /// </summary>
    public sealed class DataEngineerAgent : ArchenaAgentBase
    {
        public DataEngineerAgent(
            ArchenaKernelOrchestrator orchestrator,
            IKafkaConnector kafka,
            ILogger<DataEngineerAgent> logger)
            : base(orchestrator, kafka, logger)
        {
        }

        public override string Name => "data-engineer";

        protected override LoopOptions GetLoopOptions()
        {
            return new LoopOptions
            {
                MaxIterations = 4,
                UseMemory = true,
                UseReflection = true,
                UseCorrection = true,
                TerminationRegex = @"\b(final data model|final pipeline|etl design complete)\b"
            };
        }
        public override bool CanHandle(AgentMessageType type) =>
        type is AgentMessageType.Act
         or AgentMessageType.Request
         or AgentMessageType.Think;

        protected override string BuildPrompt(AgentEnvelope envelope)
        {
            return
$@"You are the ArchenaAI Data Engineer Agent.

You reason about:
- Data pipelines (batch and streaming)
- Data models (OLTP, OLAP, star/snowflake schemas)
- Kafka topics, message contracts, and lake/warehouse integration
- Performance, governance, and reliability of data flows

User task:
{envelope.Payload}";
        }
    }
}
