using Confluent.Kafka;
using Franz.Common.Messaging.Configuration;
using Franz.Common.Messaging.Kafka;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Text.Json;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka
{
    /// <summary>
    /// Provides a Franz-compatible Kafka connector for the ArchenaAI Semantic Kernel.
    /// </summary>
    public sealed class KafkaConnector : IKafkaConnector
    {
        private readonly IOptions<MessagingOptions> _options;
        private readonly IKafkaConsumerFactory _consumerFactory;
        private readonly ILogger<KafkaConnector> _logger;
        private readonly IProducer<string, string> _producer;

        /// <summary>
        /// Initializes a new instance of the <see cref="KafkaConnector"/> class.
        /// </summary>
        public KafkaConnector(
            IOptions<MessagingOptions> options,
            IKafkaConsumerFactory consumerFactory,
            ILogger<KafkaConnector> logger)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _consumerFactory = consumerFactory ?? throw new ArgumentNullException(nameof(consumerFactory));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));

            var producerConfig = new ProducerConfig
            {
                BootstrapServers = _options.Value.BootStrapServers,
                Acks = Acks.All,
                EnableIdempotence = true,
                LingerMs = 10,
                BatchSize = 32 * 1024
            };

            _producer = new ProducerBuilder<string, string>(producerConfig).Build();
        }

        /// <inheritdoc/>
        public async Task PublishAsync(string topic, object message, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(topic))
                throw new ArgumentException("Kafka topic cannot be null or empty.", nameof(topic));
            if (message is null)
                throw new ArgumentNullException(nameof(message));

            try
            {
                var payload = JsonSerializer.Serialize(message);

                _logger.LogDebug("[KafkaConnector] Publishing message to '{Topic}' ({Size} bytes)", topic, payload.Length);

                var result = await _producer
                    .ProduceAsync(topic, new Message<string, string> { Key = Guid.NewGuid().ToString(), Value = payload }, ct)
                    .ConfigureAwait(false);

                _logger.LogInformation("[KafkaConnector] Delivered message to {TopicPartitionOffset}", result.TopicPartitionOffset);
            }
            catch (ProduceException<string, string> ex)
            {
                _logger.LogError(ex, "[KafkaConnector] Produce error on topic '{Topic}'", topic);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task SubscribeAsync(string topic, Func<string, Task> handler, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(topic))
                throw new ArgumentException("Kafka topic cannot be null or empty.", nameof(topic));
            if (handler is null)
                throw new ArgumentNullException(nameof(handler));

            _logger.LogInformation("[KafkaConnector] Subscribing to topic '{Topic}'", topic);

            // Franz factory builds its own consumer internally
            var consumer = _consumerFactory.Build(default!);

            await Task.Run(() =>
            {
                consumer.Subscribe(topic);

                while (!ct.IsCancellationRequested)
                {
                    try
                    {
                        // Franz expects a TimeSpan poll interval instead of CancellationToken
                        var record = consumer.Consume(TimeSpan.FromMilliseconds(250));

                        if (record?.Message?.Value is null)
                            continue;

                        var message = record.Message.Value.ToString() ?? string.Empty;

                        _logger.LogDebug("[KafkaConnector] Message received from '{Topic}': {Message}", topic, message);

                        _ = Task.Run(() => handler(message), ct);
                    }
                    catch (ConsumeException ex)
                    {
                        _logger.LogError(ex, "[KafkaConnector] Consume error on topic '{Topic}'", topic);
                    }
                    catch (OperationCanceledException)
                    {
                        _logger.LogInformation("[KafkaConnector] Cancellation requested for topic '{Topic}'", topic);
                        break;
                    }
                }

                consumer.Close();
            }, ct).ConfigureAwait(false);
        }
    }
}
