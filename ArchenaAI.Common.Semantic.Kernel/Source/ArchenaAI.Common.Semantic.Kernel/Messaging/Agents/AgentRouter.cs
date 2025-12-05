using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Messaging.Agents
{
    /// <summary>
    /// Central routing layer for multi-agent message dispatch.
    /// Capable of:
    ///  - direct routing
    ///  - multi-recipient routing
    ///  - role-based routing
    ///  - capability matching (future-ready)
    /// </summary>
    public sealed class AgentRouter
    {
        private readonly ConcurrentDictionary<string, IAgent> _agents = new(StringComparer.OrdinalIgnoreCase);
        private readonly ILogger<AgentRouter> _logger;

        public AgentRouter(ILogger<AgentRouter> logger)
        {
            _logger = logger;
        }

        // ----------------------------------------------------------
        // Registration
        // ----------------------------------------------------------
        public void RegisterAgent(IAgent agent)
        {
            if (agent == null)
                throw new ArgumentNullException(nameof(agent));

            _agents[agent.Name] = agent;

            _logger.LogInformation("Registered agent: {Name}", agent.Name);
        }

        public IEnumerable<string> ListAgents() => _agents.Keys;

        // ----------------------------------------------------------
        // Core routing
        // ----------------------------------------------------------
        public async Task RouteAsync(AgentEnvelope envelope, CancellationToken ct)
        {
            if (envelope == null)
                throw new ArgumentNullException(nameof(envelope));

            // Direct routing (default)
            if (!string.IsNullOrWhiteSpace(envelope.Agent))
            {
                if (_agents.TryGetValue(envelope.Agent, out var target))
                {
                    await target.HandleAsync(envelope, ct);
                    return;
                }

                _logger.LogWarning("Agent '{Agent}' not found for envelope {CorrelationId}",
                    envelope.Agent, envelope.CorrelationId);
            }

            // If the envelope has no direct target → try type-based routing
            if (!string.IsNullOrWhiteSpace(envelope.Type.ToString()))
            {
                var candidates = _agents.Values
                    .Where(a => a.CanHandle(envelope.Type))
                    .ToList();

                if (candidates.Count == 1)
                {
                    await candidates[0].HandleAsync(envelope, ct);
                    return;
                }

                if (candidates.Count > 1)
                {
                    _logger.LogInformation(
                        "Broadcasting envelope {CorrelationId} to {Count} matching agents",
                        envelope.CorrelationId, candidates.Count);

                    foreach (var agent in candidates)
                        _ = agent.HandleAsync(envelope, ct); // fire-and-forget OK for broadcast

                    return;
                }
            }

            _logger.LogError("No routing path found for envelope {CorrelationId}", envelope.CorrelationId);
        }
    }
}
