using System;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka
{
    /// <summary>
    /// Defines a contract for producing and consuming messages in a Kafka cluster.
    /// </summary>
    /// <remarks>
    /// Implementations of this interface are responsible for managing Kafka producers and consumers,
    /// handling serialization, and ensuring delivery reliability for event-driven architectures.
    /// </remarks>
    public interface IKafkaConnector
    {
        /// <summary>
        /// Publishes a message to the specified Kafka topic.
        /// </summary>
        /// <param name="topic">
        /// The name of the Kafka topic to which the message should be published.
        /// </param>
        /// <param name="message">
        /// The message payload to send. The object should be serializable
        /// (e.g., JSON, Avro, or another supported serialization format).
        /// </param>
        /// <param name="ct">
        /// A <see cref="CancellationToken"/> used to cancel the publishing operation.
        /// </param>
        /// <returns>
        /// A <see cref="Task"/> representing the asynchronous operation.
        /// </returns>
        Task PublishAsync(string topic, object message, CancellationToken ct);

        /// <summary>
        /// Subscribes to a Kafka topic and continuously consumes messages.
        /// </summary>
        /// <param name="topic">
        /// The name of the Kafka topic to subscribe to.
        /// </param>
        /// <param name="handler">
        /// A delegate function invoked for each received message payload.
        /// The function receives the message content as a <see cref="string"/> and
        /// executes user-defined asynchronous processing logic.
        /// </param>
        /// <param name="ct">
        /// A <see cref="CancellationToken"/> used to stop consuming messages.
        /// </param>
        /// <returns>
        /// A <see cref="Task"/> representing the asynchronous subscription process.
        /// </returns>
        Task SubscribeAsync(string topic, Func<string, Task> handler, CancellationToken ct);
    }
}
