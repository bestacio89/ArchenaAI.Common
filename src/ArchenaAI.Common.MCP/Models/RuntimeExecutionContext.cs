using System;

namespace ArchenaAI.Common.MCP.Models
{
    /// <summary>
    /// Carries the ambient identity/tracing data an action was raised under.
    /// Field names mirror Kernel's ActionContext (CorrelationId, CallingAgent)
    /// on purpose, so this can be built directly from an ActionContext or
    /// SkillContext later without inventing new vocabulary.
    /// </summary>
    public sealed record RuntimeExecutionContext
    {
        public string CorrelationId { get; init; } = Guid.CreateVersion7().ToString("N");
        public string? CallingAgent { get; init; }
        public DateTimeOffset RequestedAt { get; init; } = DateTimeOffset.UtcNow;
    }
}