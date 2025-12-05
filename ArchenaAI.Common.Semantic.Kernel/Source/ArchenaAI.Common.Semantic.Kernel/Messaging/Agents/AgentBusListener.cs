using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Messaging.Agents
{
    public sealed class AgentBusListener
    {
        private readonly IKafkaConnector _kafka;
        private readonly AgentRouter _router;
        private readonly ILogger<AgentBusListener> _logger;

        public AgentBusListener(
            IKafkaConnector kafka,
            AgentRouter router,
            ILogger<AgentBusListener> logger)
        {
            _kafka = kafka;
            _router = router;
            _logger = logger;
        }

        public async Task StartAsync(string topic, CancellationToken ct)
        {
            await _kafka.SubscribeAsync(topic, async (json) =>
            {
                var envelope = AgentMessageSerializer.Deserialize(json);
                await _router.RouteAsync(envelope, ct);

            }, ct);
        }
    }
}
