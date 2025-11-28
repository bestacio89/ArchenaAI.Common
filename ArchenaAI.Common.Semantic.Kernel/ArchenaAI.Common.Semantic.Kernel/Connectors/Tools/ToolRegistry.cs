using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Tools
{
    /// <summary>
    /// Provides an in-memory registry for tools within the semantic kernel.
    /// </summary>
    /// <remarks>
    /// This class implements the <see cref="IToolsRegistry"/> interface and serves as the default
    /// in-memory implementation for managing registered tools and their associated executors.
    /// 
    /// Each registered tool includes:
    /// <list type="bullet">
    ///   <item><description>A unique tool name (case-insensitive).</description></item>
    ///   <item><description>A delegate executor responsible for executing commands asynchronously.</description></item>
    ///   <item><description>A <see cref="ToolDescriptor"/> providing metadata such as description and name.</description></item>
    /// </list>
    /// 
    /// This registry is thread-safe and optimized for concurrent access via <see cref="ConcurrentDictionary{TKey, TValue}"/>.
    /// </remarks>
    public sealed class ToolsRegistry : IToolsRegistry
    {
        private readonly ILogger<ToolsRegistry> _logger;

        /// <summary>
        /// Internal registry mapping tool names to their execution delegates and metadata.
        /// </summary>
        private readonly ConcurrentDictionary<string, (Func<string, CancellationToken, Task<string>> Executor, ToolDescriptor Meta)>
            _registry = new(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// Initializes a new instance of the <see cref="ToolsRegistry"/> class.
        /// </summary>
        /// <param name="logger">The logger instance used for diagnostic and lifecycle events.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="logger"/> is <see langword="null"/>.</exception>
        public ToolsRegistry(ILogger<ToolsRegistry> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Registers a new tool in the registry.
        /// </summary>
        /// <param name="toolName">The unique, case-insensitive name of the tool to register.</param>
        /// <param name="executor">The asynchronous function delegate responsible for executing tool commands.</param>
        /// <param name="description">A brief textual description of the tool’s purpose or functionality.</param>
        /// <returns>
        /// <see langword="true"/> if the tool was successfully registered; 
        /// otherwise, <see langword="false"/> if a tool with the same name already exists.
        /// </returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="toolName"/> is null or empty.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="executor"/> is <see langword="null"/>.</exception>
        public bool RegisterTool(string toolName, Func<string, CancellationToken, Task<string>> executor, string description)
        {
            if (string.IsNullOrWhiteSpace(toolName))
                throw new ArgumentException("Tool name cannot be null or empty.", nameof(toolName));

            ArgumentNullException.ThrowIfNull(executor);

            var meta = new ToolDescriptor(toolName, description);
            var added = _registry.TryAdd(toolName, (executor, meta));

            if (added)
                _logger.LogInformation("[ToolsRegistry] Registered new tool: {Tool}", toolName);
            else
                _logger.LogWarning("[ToolsRegistry] Tool '{Tool}' is already registered.", toolName);

            return added;
        }

        /// <summary>
        /// Unregisters a tool from the registry.
        /// </summary>
        /// <param name="toolName">The name of the tool to remove.</param>
        /// <returns>
        /// <see langword="true"/> if the tool was successfully unregistered;
        /// otherwise, <see langword="false"/> if it was not found.
        /// </returns>
        public bool UnregisterTool(string toolName)
        {
            var removed = _registry.TryRemove(toolName, out _);
            if (removed)
                _logger.LogInformation("[ToolsRegistry] Unregistered tool: {Tool}", toolName);
            else
                _logger.LogWarning("[ToolsRegistry] Tool '{Tool}' not found for unregistration.", toolName);

            return removed;
        }

        /// <summary>
        /// Determines whether a tool with the specified name is registered.
        /// </summary>
        /// <param name="toolName">The name of the tool to check.</param>
        /// <returns>
        /// <see langword="true"/> if the tool exists in the registry; otherwise, <see langword="false"/>.
        /// </returns>
        public bool ContainsTool(string toolName)
            => _registry.ContainsKey(toolName);

        /// <summary>
        /// Retrieves the executor delegate for a registered tool.
        /// </summary>
        /// <param name="toolName">The name of the tool to retrieve.</param>
        /// <returns>
        /// A <see cref="Func{T1, T2, TResult}"/> representing the asynchronous executor delegate,
        /// or <see langword="null"/> if the tool is not registered.
        /// </returns>
        public Func<string, CancellationToken, Task<string>>? GetExecutor(string toolName)
            => _registry.TryGetValue(toolName, out var entry) ? entry.Executor : null;

        /// <summary>
        /// Lists all tools currently registered in the registry.
        /// </summary>
        /// <returns>
        /// An immutable collection of <see cref="ToolDescriptor"/> instances describing the available tools.
        /// </returns>
        public IReadOnlyCollection<ToolDescriptor> ListTools()
            => _registry.Values.Select(v => v.Meta).ToList();
    }
}
