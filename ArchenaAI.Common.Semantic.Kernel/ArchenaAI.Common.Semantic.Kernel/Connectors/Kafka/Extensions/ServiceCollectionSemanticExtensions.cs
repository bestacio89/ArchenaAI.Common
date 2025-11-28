using ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka;
using ArchenaAI.Common.Semantic.Kernel.Messaging;
using Franz.Common.DependencyInjection.Extensions;
using Franz.Common.Messaging.Kafka.Extensions;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka.Extensions
{
    /// <summary>
    /// Provides extension methods for integrating Franz messaging with the ArchenaAI Semantic Kernel.
    /// </summary>
    public static class ServiceCollectionSemanticExtensions
    {
        /// <summary>
        /// Adds Franz Kafka messaging infrastructure and semantic-layer integration
        /// for the ArchenaAI Semantic Kernel.
        /// </summary>
        /// <param name="services">The current <see cref="IServiceCollection"/>.</param>
        /// <param name="configuration">The application configuration containing Kafka settings.</param>
        /// <returns>The updated <see cref="IServiceCollection"/> instance.</returns>
        public static IServiceCollection AddSemanticMessaging(this IServiceCollection services, IConfiguration configuration)
        {
            // ✅ Register Franz Kafka foundation (producers, consumers, models)
            services.AddMessaging(configuration);

            // ✅ Add the semantic Kafka connector (high-level adapter)
            services.AddNoDuplicateScoped<IKafkaConnector, KafkaConnector>();

            // ✅ Add semantic envelope serialization utilities
            services.AddNoDuplicateSingleton<ISemanticMessageSerializer, SemanticMessageSerializer>();

            // ✅ Add logger for observability
            services.AddLogging(builder =>
            {
                builder.AddFilter("ArchenaAI.Common.Semantic.Kernel.Connectors.Kafka", LogLevel.Information);
            });

            return services;
        }
    }
}
