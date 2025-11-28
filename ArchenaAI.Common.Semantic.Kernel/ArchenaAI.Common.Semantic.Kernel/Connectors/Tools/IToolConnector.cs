using System.Threading;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Tools
{
    /// <summary>
    /// Defines a contract for executing external tools or utility commands
    /// within the ArchenaAI Semantic Kernel environment.
    /// </summary>
    /// <remarks>
    /// The <see cref="IToolsConnector"/> interface enables semantic agents or orchestrators
    /// to invoke and interact with external system tools (e.g., <c>curl</c>, <c>python</c>, <c>ffmpeg</c>)
    /// in a controlled, sandboxed manner.
    /// </remarks>
    public interface IToolsConnector
    {
        /// <summary>
        /// Executes a tool command asynchronously and returns its standard output.
        /// </summary>
        /// <param name="toolName">
        /// The name or path of the executable tool (for example: <c>"python"</c>, <c>"ffmpeg"</c>, or <c>"powershell"</c>).
        /// </param>
        /// <param name="command">
        /// The arguments or subcommand to pass to the tool (for example: <c>"--version"</c> or <c>"script.py arg1"</c>).
        /// </param>
        /// <param name="ct">
        /// A <see cref="CancellationToken"/> used to cancel execution if required.
        /// </param>
        /// <returns>
        /// A <see cref="Task{TResult}"/> representing the asynchronous operation.
        /// The task result contains the tool’s captured standard output as a <see cref="string"/>.
        /// </returns>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="toolName"/> or <paramref name="command"/> is null or empty.
        /// </exception>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the tool execution fails or returns a non-zero exit code.
        /// </exception>
        Task<string> ExecuteAsync(string toolName, string command, CancellationToken ct);
    }
}
