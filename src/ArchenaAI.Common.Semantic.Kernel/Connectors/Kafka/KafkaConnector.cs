using Confluent.Kafka;
using Franz.Common.Messaging.Kafka;
using Franz.Common.Messaging.Kafka.Configuration;
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
        private readonly IOptions<KafkaMessagingOptions> _options;
        private readonly IKafkaConsumerFactory _consumerFactory;
        private readonly ILogger<KafkaConnector> _logger;
        private readonly IProducer<string, string> _producer;

        /// <summary>
        /// Initializes a new instance of the <see cref="KafkaConnector" /> class.
        /// </summary>
        public KafkaConnector(
            IOptions<KafkaMessagingOptions> options,
            IKafkaConsumerFactory consumerFactory,
            ILogger<KafkaConnector> logger)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _consumerFactory = consumerFactory ?? throw new ArgumentNullException(nameof(consumerFactory));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));

            var kafkaOptions = _options.Value;
            var producerConfig = new ProducerConfig
            {
                BootstrapServers = kafkaOptions.BootstrapServers,
                ClientId = kafkaOptions.ClientId,
                Acks = kafkaOptions.Producer.Acks switch
                {
                    KafkaAcks.None => Acks.None,
                    KafkaAcks.Leader => Acks.Leader,
                    _ => Acks.All
                },
                EnableIdempotence = kafkaOptions.Producer.EnableIdempotence,
                LingerMs = kafkaOptions.Producer.LingerMs,
                BatchSize = kafkaOptions.Producer.BatchSize,
                MessageMaxBytes = kafkaOptions.Producer.MessageMaxBytes,
                MessageSendMaxRetries = kafkaOptions.Producer.MessageSendMaxRetries,
                RetryBackoffMs = kafkaOptions.Producer.RetryBackoffMs,
                CompressionType = kafkaOptions.Producer.CompressionType switch
                {
                    KafkaCompressionType.Gzip => Confluent.Kafka.CompressionType.Gzip,
                    KafkaCompressionType.Lz4 => Confluent.Kafka.CompressionType.Lz4,
                    KafkaCompressionType.Zstd => Confluent.Kafka.CompressionType.Zstd,
                    KafkaCompressionType.None => Confluent.Kafka.CompressionType.None,
                    _ => Confluent.Kafka.CompressionType.Snappy
                },
                SecurityProtocol = kafkaOptions.Security.SecurityProtocol switch
                {
                    KafkaSecurityProtocol.Ssl => Confluent.Kafka.SecurityProtocol.Ssl,
                    KafkaSecurityProtocol.SaslPlaintext => Confluent.Kafka.SecurityProtocol.SaslPlaintext,
                    KafkaSecurityProtocol.SaslSsl => Confluent.Kafka.SecurityProtocol.SaslSsl,
                    _ => Confluent.Kafka.SecurityProtocol.Plaintext
                },
                SaslMechanism = kafkaOptions.Security.SaslMechanism switch
                {
                    KafkaSaslMechanism.Plain => Confluent.Kafka.SaslMechanism.Plain,
                    KafkaSaslMechanism.ScramSha256 => Confluent.Kafka.SaslMechanism.ScramSha256,
                    KafkaSaslMechanism.ScramSha512 => Confluent.Kafka.SaslMechanism.ScramSha512,
                    KafkaSaslMechanism.Gssapi => Confluent.Kafka.SaslMechanism.Gssapi,
                    KafkaSaslMechanism.OAuthBearer => Confluent.Kafka.SaslMechanism.OAuthBearer,
                    _ => null
                },
                SaslUsername = kafkaOptions.Security.SaslUsername,
                SaslPassword = kafkaOptions.Security.SaslPassword,
                SslCaLocation = kafkaOptions.Security.SslCaLocation,
                SslCertificateLocation = kafkaOptions.Security.SslCertificateLocation,
                SslKeyLocation = kafkaOptions.Security.SslKeyLocation
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

            // Franz owns consumer configuration; its factory takes no arguments.
            var consumer = _consumerFactory.Build();

            await Task.Run(() =>
            {
                consumer.Subscribe(topic);

                while (!ct.IsCancellationRequested)
                {
                    try
                    {
                        var record = consumer.Consume(TimeSpan.FromMilliseconds(250));

                        if (record?.Message?.Value is null)
                            continue;

                        var message = record.Message.Value;

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
