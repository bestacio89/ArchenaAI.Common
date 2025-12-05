using ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka;
using ArchenaAI.Common.Semantic.Kernel.Messaging;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel
{
    /// <summary>
    /// Coordinates the emission of semantic events across the distributed AI kernel.
    /// </summary>
    /// <remarks>
    /// The <see cref="SemanticOrchestrator"/> acts as a high-level semantic event bus publisher.
    /// It leverages the <see cref="IKafkaConnector"/> and <see cref="ISemanticMessageSerializer"/>
    /// to send structured semantic messages between agents, memory services, and reasoning components.
    /// </remarks>
    public sealed class SemanticOrchestrator
    {
        private readonly IKafkaConnector _kafka;
        private readonly ISemanticMessageSerializer _serializer;
        private readonly ILogger<SemanticOrchestrator> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="SemanticOrchestrator"/> class.
        /// </summary>
        /// <param name="kafka">The Kafka connector responsible for publishing semantic events.</param>
        /// <param name="serializer">The serializer used to transform semantic messages into JSON.</param>
        /// <param name="logger">The logger instance for observability and diagnostics.</param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="kafka"/>, <paramref name="serializer"/>, or <paramref name="logger"/> is <see langword="null"/>.
        /// </exception>
        public SemanticOrchestrator(
            IKafkaConnector kafka,
            ISemanticMessageSerializer serializer,
            ILogger<SemanticOrchestrator> logger)
        {
            _kafka = kafka ?? throw new ArgumentNullException(nameof(kafka));
            _serializer = serializer ?? throw new ArgumentNullException(nameof(serializer));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Emits a reasoning or skill-invocation event to the semantic message bus.
        /// </summary>
        /// <param name="skill">The name of the skill or reasoning process that triggered the event.</param>
        /// <param name="content">The serialized content or contextual payload of the event.</param>
        /// <param name="ct">A <see cref="CancellationToken"/> used to cancel the operation.</param>
        /// <returns>A <see cref="Task"/> representing the asynchronous publishing operation.</returns>
        public async Task EmitReasoningEventAsync(string skill, string content, CancellationToken ct)
        {
            var message = new SemanticMessage
            {
                Type = "skill.invocation",
                Content = content,
                Metadata = new Dictionary<string, object?>
                {
                    ["skill"] = skill,
                    ["context"] = "semantic.kernel"
                }
            };

            _logger.LogDebug("[SemanticOrchestrator] Emitting event for skill '{Skill}'", skill);

            await _kafka
                .PublishAsync("archenaai.semantic.events", _serializer.Serialize(message), ct)
                .ConfigureAwait(false);

            _logger.LogInformation("[SemanticOrchestrator] Event emitted successfully for skill '{Skill}'", skill);
        }
    }
}
