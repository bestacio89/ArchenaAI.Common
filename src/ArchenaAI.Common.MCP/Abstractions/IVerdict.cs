using System.Collections.Generic;

namespace ArchenaAI.Common.MCP.Abstractions;

public interface IVerdict
{
    bool IsAllowed { get; }
    GovernanceSeverity MaxSeverity { get; }
    IReadOnlyCollection<string> ViolatedRuleIds { get; }
}
