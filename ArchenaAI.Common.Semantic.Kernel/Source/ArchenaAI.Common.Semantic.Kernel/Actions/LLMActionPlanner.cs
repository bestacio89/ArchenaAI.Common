using System;
using System.Text.RegularExpressions;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Actions;
using ArchenaAI.Common.Semantic.Kernel.Skills;

namespace ArchenaAI.Common.Semantic.Kernel.Planning
{
    /// <summary>
    /// Extracts tool/action calls from LLM output and executes 
    /// them through the ActionRegistry.
    /// </summary>
    public sealed class LLMActionPlanner
    {
        private readonly ActionRegistry _actions;
        private readonly IArchenaKernel _kernel;

        private static readonly Regex ActionRegex =
            new(@"<Action name=""(?<name>.*?)"">\s*(?<json>\{.*?\})\s*</Action>",
                RegexOptions.Singleline | RegexOptions.IgnoreCase);

        public LLMActionPlanner(ActionRegistry actions, IArchenaKernel kernel)
        {
            _actions = actions;
            _kernel = kernel;
        }

        public async Task<string> ProcessAsync(
            string llmResponse,
            string correlationId,
            CancellationToken ct)
        {
            var match = ActionRegex.Match(llmResponse);
            if (!match.Success)
                return llmResponse; // No tool call

            var name = match.Groups["name"].Value;
            var json = match.Groups["json"].Value;

            var args = JsonSerializer.Deserialize<Dictionary<string, object?>>(json)
                       ?? new();

            var ctx = new ActionContext(
                input: llmResponse,
                args,
                caller: "llm",
                correlationId);

            if (!_actions.TryGet(name, out var action))
                return $"[Action '{name}' not found]";

            var result = await action.ExecuteAsync(ctx, ct);

            return result?.ToString() ?? string.Empty;
        }
    }
}
