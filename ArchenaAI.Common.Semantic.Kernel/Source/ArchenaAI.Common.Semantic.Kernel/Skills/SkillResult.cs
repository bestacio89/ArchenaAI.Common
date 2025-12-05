using System;
using System.Collections.Generic;

namespace ArchenaAI.Common.Semantic.Kernel.Skills;

public sealed class SkillResult
{
    public bool Success { get; }
    public object? Output { get; }
    public Exception? Error { get; }
    public IReadOnlyDictionary<string, object?> Metadata { get; }

    private SkillResult(
        bool success,
        object? output,
        Exception? error,
        IReadOnlyDictionary<string, object?> metadata)
    {
        Success = success;
        Output = output;
        Error = error;
        Metadata = metadata;
    }

    public static SkillResult FromSuccess(
        object? output,
        IReadOnlyDictionary<string, object?>? metadata = null)
        => new(true, output, null, metadata ?? new Dictionary<string, object?>());

    public static SkillResult FromError(
        Exception error,
        IReadOnlyDictionary<string, object?>? metadata = null)
        => new(false, null, error, metadata ?? new Dictionary<string, object?>());
}
