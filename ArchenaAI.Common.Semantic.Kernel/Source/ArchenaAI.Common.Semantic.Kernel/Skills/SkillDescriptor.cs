using System;
using System.Collections.Generic;

namespace ArchenaAI.Common.Semantic.Kernel.Skills;

public sealed class SkillDescriptor
{
    public string Name { get; }
    public string? Description { get; }
    public SkillCategory Category { get; }
    public IReadOnlyCollection<string> Tags { get; }

    /// <summary>
    /// Optional .NET type representing the expected input payload.
    /// This can be used for validation / serialization.
    /// </summary>
    public Type? InputType { get; }

    /// <summary>
    /// Optional .NET type representing the output payload.
    /// </summary>
    public Type? OutputType { get; }

    /// <summary>
    /// Optional list of model identifiers this skill is allowed to use.
    /// (For LLM-based skills only.)
    /// </summary>
    public IReadOnlyCollection<string> AllowedModels { get; }

    public SkillDescriptor(
        string name,
        string? description = null,
        SkillCategory category = SkillCategory.Unknown,
        IEnumerable<string>? tags = null,
        Type? inputType = null,
        Type? outputType = null,
        IEnumerable<string>? allowedModels = null)
    {
        Name = string.IsNullOrWhiteSpace(name)
            ? throw new ArgumentException("Skill name cannot be null or empty.", nameof(name))
            : name;

        Description = description;
        Category = category;
        Tags = tags is null ? Array.Empty<string>() : new List<string>(tags);
        InputType = inputType;
        OutputType = outputType;
        AllowedModels = allowedModels is null ? Array.Empty<string>() : new List<string>(allowedModels);
    }

    public override string ToString() => $"{Name} ({Category})";
}
