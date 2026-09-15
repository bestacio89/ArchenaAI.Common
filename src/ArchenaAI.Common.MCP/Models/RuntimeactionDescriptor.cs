using ArchenaAI.Common.MCP.Abstractions;
using System.Collections.Generic;

namespace ArchenaAI.Common.MCP.Models
{
    public sealed record RuntimeActionDescriptor : IRuntimeActionDescriptor
    {
        public string ActionId { get; init; }
        public string OriginComponent { get; init; }
        public string TargetComponent { get; init; }
        public string OriginLayer { get; init; }
        public string TargetLayer { get; init; }
        public string CapabilityId { get; init; }
        public IReadOnlySet<string> ResponsibilityDomains { get; init; }

        // Was "ExecutionContext" — that name resolves to System.Threading's
        // type when nothing local shadows it, which is not what was intended.
        public RuntimeExecutionContext Context { get; init; } = new();
    }
}