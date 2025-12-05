using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Tools
{
    /// <summary>
    /// Defines a contract for a centralized registry that manages all available tool executors.
    /// </summary>
    /// <remarks>
    /// The <see cref="IToolsRegistry"/> provides a discovery mechanism for semantic agents,
    /// allowing them to dynamically explore, register, or remove executors at runtime.
    /// </remarks>
    public interface IToolsRegistry
    {
        /// <summary>
        /// Registers a new tool executor into the registry.
        /// </summary>
        /// <param name="toolName">The name of the tool (e.g., "terraform", "bash").</param>
        /// <param name="executor">The function that executes the tool asynchronously.</param>
        /// <param name="description">A brief description of the tool’s purpose and usage.</param>
        /// <returns><c>true</c> if registration succeeded; otherwise, <c>false</c>.</returns>
        bool RegisterTool(string toolName, Func<string, CancellationToken, Task<string>> executor, string description);

        /// <summary>
        /// Unregisters an existing tool from the registry.
        /// </summary>
        /// <param name="toolName">The name of the tool to remove.</param>
        /// <returns><c>true</c> if the tool was successfully removed; otherwise, <c>false</c>.</returns>
        bool UnregisterTool(string toolName);

        /// <summary>
        /// Determines whether a specific tool is registered.
        /// </summary>
        /// <param name="toolName">The tool name to check.</param>
        /// <returns><c>true</c> if the tool is registered; otherwise, <c>false</c>.</returns>
        bool ContainsTool(string toolName);

        /// <summary>
        /// Retrieves the delegate for a registered tool executor.
        /// </summary>
        /// <param name="toolName">The name of the registered tool.</param>
        /// <returns>The executor function if found; otherwise, <c>null</c>.</returns>
        Func<string, CancellationToken, Task<string>>? GetExecutor(string toolName);

        /// <summary>
        /// Lists all registered tools and their metadata.
        /// </summary>
        /// <returns>A collection of registered tool descriptors.</returns>
        IReadOnlyCollection<ToolDescriptor> ListTools();
    }
}
