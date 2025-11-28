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
    /// Memory Indexer Agent:
    /// Responsible for semantic/symbolic memory indexing, normalization,
    /// tagging, deduplication and long-term memory summarization.
    /// </summary>
    public sealed class MemoryIndexerAgent : ArchenaAgentBase
    {
        public MemoryIndexerAgent(
            ArchenaKernelOrchestrator orchestrator,
            IKafkaConnector kafka,
            ILogger<MemoryIndexerAgent> logger)
            : base(orchestrator, kafka, logger)
        {
        }

        public override string Name => "memory.indexer";

        /// <summary>
        /// Memory indexing tasks rarely need long chains.
        /// Reflection + summarization are useful, but not full planning.
        /// </summary>
        protected override LoopOptions GetLoopOptions() =>
            new LoopOptions
            {
                MaxIterations = 3,
                UseReflection = true,
                UseCorrection = true,
                UseMemory = false, // MemoryIndexer writes memory, does not read it unless needed externally
                TerminationRegex = "^(done|complete|indexed)$"
            };


        public override bool CanHandle(AgentMessageType type) =>
         type is AgentMessageType.MemoryIndex
         or AgentMessageType.MemoryStore
         or AgentMessageType.MemorySearch
         or AgentMessageType.Normalize;

        /// <summary>
        /// Memory indexer prompt builder.
        /// This wraps incoming content into instructions for normalization,
        /// tagging, duplication detection, semantic structuring, and summarization.
        /// </summary>
        protected override string BuildPrompt(AgentEnvelope envelope)
        {
            var content = envelope.Payload ?? string.Empty;

            return $@"
You are the Memory Indexer Agent of ArchenaAI.
Your job is STRICTLY to:

1. Normalize the text
2. Extract meaning and structure
3. Identify important metadata and topics
4. Detect duplication patterns
5. Prepare the content for long-term storage
6. Produce a clean, machine-optimized version of the content

NEVER hallucinate.
NEVER expand the content.
NEVER rewrite creatively.
ONLY CLEAN, COMPRESS, TAG, STRUCTURE.

Return output in this JSON structure:

{{
  ""normalized"": ""clean text..."",
  ""topics"": [""tag1"", ""tag2""],
  ""summary"": ""short compression"",
  ""deduplicationHash"": ""hash-of-meaning"",
  ""metadata"": {{
      ""sourceAgent"": ""{envelope.Agent}"",
      ""correlationId"": ""{envelope.CorrelationId}""
  }}
}}

CONTENT TO INDEX:
-----------------
{content}
";
        }
    }
}
