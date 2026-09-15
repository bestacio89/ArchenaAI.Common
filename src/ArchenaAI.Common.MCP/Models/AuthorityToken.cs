using ArchenaAI.Common.MCP.Abstractions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ArchenaAI.Common.MCP.Models
{
    public sealed record AuthorityToken : IAuthorityToken
    {
        public string TokenId { get; init; }
        public string IssuedTo { get; init; }
        public DateTimeOffset IssuedAt { get; init; }
        public DateTimeOffset ExpiresAt { get; init; }
        public bool Revocable { get; init; }
        public IReadOnlyCollection<Capability> Capabilities { get; init; } = Array.Empty<Capability>();

        public bool HasCapability(string capabilityId) =>
            Capabilities.Any(c => c.Id == capabilityId);
    }
}