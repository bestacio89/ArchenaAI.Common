using System.Collections.Generic;
using ArchenaAI.Common.MCP.Models;

namespace ArchenaAI.Common.MCP.Abstractions
{
    public interface IRuntimeActionDescriptor
    {
        string ActionId { get; }           // "repo.read", "db.write"
        string OriginComponent { get; }    // Kernel / Runtime / Agent
        string TargetComponent { get; }    // DB / FS / API
        string OriginLayer { get; }        // Api / App / Domain / Infra
        string TargetLayer { get; }
        string CapabilityId { get; }
        IReadOnlySet<string> ResponsibilityDomains { get; }
        RuntimeExecutionContext Context { get; }
    }
}