using System;

namespace ArchenaAI.Common.Semantic.Kernel.Skills;

[AttributeUsage(AttributeTargets.Class, Inherited = false, AllowMultiple = false)]
public sealed class SkillAttribute : Attribute
{
    public string Name { get; }
    public string? Description { get; init; }
    public SkillCategory Category { get; init; } = SkillCategory.Unknown;
    public string[] Tags { get; init; } = Array.Empty<string>();
    public Type? InputType { get; init; }
    public Type? OutputType { get; init; }
    public string[] AllowedModels { get; init; } = Array.Empty<string>();

    public SkillAttribute(string name)
    {
        Name = string.IsNullOrWhiteSpace(name)
            ? throw new ArgumentException("Skill name cannot be null or empty.", nameof(name))
            : name;
    }
}
