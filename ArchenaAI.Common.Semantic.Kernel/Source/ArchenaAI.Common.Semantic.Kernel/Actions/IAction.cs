using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Abstractions;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Actions
{
    /// <summary>
    /// Represents an executable tool/action available to agents.
    /// Deterministic, pipeline-governed alternative to LLM "function calling".
    /// </summary>
    public interface IAction
    {
        string Name { get; }

        ActionDescriptor Descriptor { get; }

        Task<object?> ExecuteAsync(IActionContext context, CancellationToken ct);
    }
}
