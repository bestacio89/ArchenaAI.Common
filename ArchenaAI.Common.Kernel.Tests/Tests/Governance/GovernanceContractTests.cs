using ArchenaAI.Common.MCP.Abstractions;
using ArchenaAI.Common.MCP.Models;
using ArchenaAI.Common.MCP.Policies;
using Xunit;

namespace ArchenaAI.Common.Kernel.Tests.Tests.Governance;

public sealed class GovernanceContractTests
{
    [Fact]
    public void EvaluationResult_DeniesOnlyViolationsAtOrAboveThreshold()
    {
        var result = new GovernanceEvaluationResult
        {
            Source = "test",
            Target = "action"
        };

        result.AddRuleResult(new GovernanceRuleResult
        {
            RuleId = "test.warning",
            RuleName = "Warning",
            Severity = GovernanceSeverity.High,
            IsCompliant = false,
            Message = "A high-severity violation."
        });

        Assert.True(result.IsAllowed(GovernanceSeverity.Critical));
        Assert.False(result.IsAllowed(GovernanceSeverity.High));
    }

    [Fact]
    public void CapabilityAuthorizationEvaluator_RejectsAuthorityWithoutRequiredGrant()
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
            TokenId = "test-token",
            IssuedTo = "test-agent",
            IssuedAt = DateTimeOffset.UtcNow,
            ExpiresAt = DateTimeOffset.UtcNow.AddMinutes(5),
            Revocable = true,
            Capabilities = Array.Empty<Capability>()
        };
        var result = new GovernanceEvaluationResult
        {
            Source = "test",
            Target = action.ActionId
        };

        new CapabilityAuthorizationEvaluator().Evaluate(
            action,
            authority,
            ExecutionBudget.ReadOnly(),
            result);

        var rule = Assert.Single(result.RuleResults);
        Assert.False(rule.IsCompliant);
        Assert.Equal("AEG-SEC-AUTH001", rule.RuleId);
        Assert.Equal(GovernanceSeverity.Critical, rule.Severity);
    }

    [Fact]
    public void CapabilityAuthorizationEvaluator_RejectsExpiredAuthorityEvenWhenGrantExists()
    {
        var action = new RuntimeActionDescriptor
        {
            ActionId = "repository.read",
            OriginComponent = "Runtime",
            TargetComponent = "Repository",
            OriginLayer = "Application",
            TargetLayer = "Infrastructure",
            CapabilityId = "repo.read",
            ResponsibilityDomains = new HashSet<string> { "Architecture" }
        };
        var authority = new AuthorityToken
        {
            TokenId = "expired-token",
            IssuedTo = "test-agent",
            IssuedAt = DateTimeOffset.UtcNow.AddMinutes(-10),
            ExpiresAt = DateTimeOffset.UtcNow.AddMinutes(-1),
            Revocable = true,
            Capabilities = new[] { Capabilities.ReadRepository() }
        };
        var result = new GovernanceEvaluationResult
        {
            Source = "test",
            Target = action.ActionId
        };

        new CapabilityAuthorizationEvaluator().Evaluate(
            action,
            authority,
            ExecutionBudget.ReadOnly(),
            result);

        Assert.False(Assert.Single(result.RuleResults).IsCompliant);
    }
}
