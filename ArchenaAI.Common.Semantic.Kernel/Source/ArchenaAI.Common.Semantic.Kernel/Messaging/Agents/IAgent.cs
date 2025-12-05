using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Messaging.Agents
{
    public interface IAgent
    {
        /// <summary>
        /// Unique agent identifier (used for routing and metadata).
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Determines if this agent can handle a given envelope message type.
        /// Used by the router for multi-agent dispatch.
        /// </summary>
        bool CanHandle(AgentMessageType type);

        /// <summary>
        /// Handles an envelope asynchronously.
        /// </summary>
        Task HandleAsync(AgentEnvelope envelope, CancellationToken ct);
    }
}
