using System;
using System.Collections.Generic;
using ArchenaAI.Common.MCP.Models;

namespace ArchenaAI.Common.MCP.Abstractions
{
    public interface IAuthorityToken
    {
        string TokenId { get; }
        string IssuedTo { get; }
        DateTimeOffset IssuedAt { get; }
        DateTimeOffset ExpiresAt { get; }
        bool Revocable { get; }
        IReadOnlyCollection<Capability> Capabilities { get; }

        bool HasCapability(string capabilityId);
    }
}