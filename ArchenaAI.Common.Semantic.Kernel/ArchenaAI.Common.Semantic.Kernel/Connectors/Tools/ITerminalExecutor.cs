using System.Diagnostics;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Tools
{
    /// <summary>
    /// Defines a contract for executing shell or terminal commands in a controlled environment.
    /// </summary>
    /// <remarks>
    /// This interface abstracts command execution, making it adaptable for different operating systems (Windows, Linux, macOS)
    /// and compatible with the semantic kernel pipeline — allowing AI agents or orchestrators to safely invoke system-level operations.
    /// </remarks>
    public interface ITerminalExecutor
    {
        /// <summary>
        /// Executes a shell or terminal command asynchronously and returns the captured output.
        /// </summary>
        /// <param name="command">
        /// The full command string to execute (e.g., <c>"ls -la"</c> or <c>"dir"</c>).
        /// </param>
        /// <param name="workingDirectory">
        /// Optional working directory where the command will be executed. If null, the current process directory is used.
        /// </param>
        /// <param name="ct">
        /// A <see cref="CancellationToken"/> used to cancel execution.
        /// </param>
        /// <returns>
        /// A <see cref="Task{TResult}"/> that represents the asynchronous operation.
        /// The task result contains the standard output and error streams of the executed command.
        /// </returns>
        Task<(string Output, string Error, int ExitCode)> ExecuteAsync(string command, string? workingDirectory, CancellationToken ct);

        /// <summary>
        /// Starts a command as a long-running background process.
        /// </summary>
        /// <param name="command">The command line to execute.</param>
        /// <param name="workingDirectory">Optional working directory for execution.</param>
        /// <returns>
        /// A <see cref="Process"/> instance representing the running process.
        /// </returns>
        Process StartDetached(string command, string? workingDirectory = null);
    }
}
