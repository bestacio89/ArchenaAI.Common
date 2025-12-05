using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Actions
{
    public sealed class ActionRegistry
    {
        private readonly ConcurrentDictionary<string, IAction> _actions =
            new(StringComparer.OrdinalIgnoreCase);

        public void Register(IAction action)
        {
            if (action == null)
                throw new ArgumentNullException(nameof(action));

            _actions[action.Name] = action;
        }

        public bool TryGet(string name, out IAction? action)
            => _actions.TryGetValue(name, out action);

        public IEnumerable<ActionDescriptor> DescribeAll()
        {
            foreach (var a in _actions.Values)
                yield return a.Descriptor;
        }

        public async Task<object?> ExecuteAsync(
            string name,
            IActionContext ctx,
            CancellationToken ct)
        {
            if (!_actions.TryGetValue(name, out var action))
                throw new KeyNotFoundException($"Action '{name}' not registered.");

            return await action.ExecuteAsync(ctx, ct);
        }
    }
}
