using ArchenaAI.Common.MCP.Models;
using ArchenaAI.Common.MCP.RuntimeProfiles;
using ArchenaAI.Common.MCP.Tribunals;
using Xunit;

namespace ArchenaAI.Common.Kernel.Tests.Tests.Governance;

public sealed class RuntimeProfileAuthorizationTests
{
    [Fact]
    public void Tribunal_RejectsCapabilityOutsideRuntimeProfileAllowList()
    {
        var action = new RuntimeActionDescriptor
        {
            ActionId = "database.write",
            OriginComponent = "Runtime",
            TargetComponent = "Database",
            OriginLayer = "Application",
            TargetLayer = "Infrastructure",
            CapabilityId = "db.write.transactional",
            ResponsibilityDomains = new HashSet<string> { "Persistence" }
        };
        var authority = new AuthorityToken
        {
            TokenId = "token-with-write-grant",
            IssuedTo = "test-agent",
            IssuedAt = DateTimeOffset.UtcNow,
            ExpiresAt = DateTimeOffset.UtcNow.AddMinutes(5),
            Revocable = true,
            Capabilities = new[] { Capabilities.DbWriteTransactional() }
        };
        var tribunal = new RuntimeTribunal([
            new ArchenaAI.Common.MCP.Policies.CapabilityAuthorizationEvaluator(),
            new ArchenaAI.Common.MCP.Policies.BoundaryComplianceEvaluator()
        ]);

        var result = tribunal.Evaluate(action, authority, AnalysisRuntime.Instance);

        Assert.False(result.IsAllowed(AnalysisRuntime.Instance.DenyThreshold));
        Assert.Contains(result.RuleResults, rule =>
            rule.RuleId == "AEG-SEC-PROFILE001" && !rule.IsCompliant);
    }
}
