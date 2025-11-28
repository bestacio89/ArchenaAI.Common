using Microsoft.Extensions.Logging;
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Tools
{
    /// <summary>
    /// Provides a unified abstraction for executing external system tools
    /// (CLI commands, scripts, or custom executors) within the semantic kernel environment.
    /// </summary>
    /// <remarks>
    /// The <see cref="ToolsConnector"/> acts as a central execution gateway for agent tools.
    /// It supports direct system commands (e.g. bash, PowerShell, Python)
    /// and registered custom executors (e.g. Terraform, Kubernetes, Docker).
    /// </remarks>
    public sealed class ToolsConnector : IToolsConnector
    {
        private readonly ILogger<ToolsConnector> _logger;

        // Registry of specialized tool executors (e.g., "terraform" -> TerraformExecutor)
        private readonly ConcurrentDictionary<string, Func<string, CancellationToken, Task<string>>> _executors;

        /// <summary>
        /// Initializes a new instance of the <see cref="ToolsConnector"/> class.
        /// </summary>
        /// <param name="logger">The logger instance used for structured diagnostics.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="logger"/> is null.</exception>
        public ToolsConnector(ILogger<ToolsConnector> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _executors = new ConcurrentDictionary<string, Func<string, CancellationToken, Task<string>>>(StringComparer.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Executes a tool asynchronously, resolving specialized executors when available.
        /// </summary>
        /// <param name="toolName">The name of the tool to execute (e.g. "terraform", "bash", "python").</param>
        /// <param name="command">The command or arguments to execute.</param>
        /// <param name="ct">A cancellation token to abort execution if needed.</param>
        /// <returns>The combined output (stdout + stderr) as a string.</returns>
        public async Task<string> ExecuteAsync(string toolName, string command, CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(toolName))
                throw new ArgumentException("Tool name cannot be null or empty.", nameof(toolName));

            if (string.IsNullOrWhiteSpace(command))
                throw new ArgumentException("Command cannot be null or empty.", nameof(command));

            _logger.LogInformation("[Tools] Executing tool '{Tool}' with command: {Command}", toolName, command);

            // Check if there's a custom executor registered for this tool
            if (_executors.TryGetValue(toolName, out var executor))
            {
                _logger.LogDebug("[Tools] Found registered executor for tool '{Tool}'", toolName);
                return await executor(command, ct).ConfigureAwait(false);
            }

            // Default fallback: execute the command via system shell
            var shell = GetSystemShell();
            var shellArgs = GetShellArguments(toolName, command);

            _logger.LogDebug("[Tools] Using fallback shell executor: {Shell} {Args}", shell, shellArgs);

            var psi = new ProcessStartInfo
            {
                FileName = shell,
                Arguments = shellArgs,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            var outputBuilder = new StringBuilder();
            var errorBuilder = new StringBuilder();

            using var process = new Process { StartInfo = psi, EnableRaisingEvents = true };

            process.OutputDataReceived += (_, e) => { if (e.Data != null) outputBuilder.AppendLine(e.Data); };
            process.ErrorDataReceived += (_, e) => { if (e.Data != null) errorBuilder.AppendLine(e.Data); };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            await process.WaitForExitAsync(ct).ConfigureAwait(false);

            var output = outputBuilder.ToString().Trim();
            var error = errorBuilder.ToString().Trim();

            if (!string.IsNullOrEmpty(error))
            {
                _logger.LogWarning("[Tools] Execution produced error output: {Error}", error);
            }

            _logger.LogInformation("[Tools] Execution completed for tool '{Tool}'", toolName);

            return string.IsNullOrEmpty(output) ? error : output;
        }

        /// <summary>
        /// Registers a custom executor for a specific tool.
        /// </summary>
        /// <param name="toolName">The name of the tool (e.g., "terraform").</param>
        /// <param name="executor">The asynchronous delegate that executes the tool command.</param>
        /// <returns><c>true</c> if registration succeeded, <c>false</c> otherwise.</returns>
        public bool RegisterExecutor(string toolName, Func<string, CancellationToken, Task<string>> executor)
        {
            ArgumentNullException.ThrowIfNull(executor);
            return _executors.TryAdd(toolName, executor);
        }

        private static string GetSystemShell()
        {
            if (OperatingSystem.IsWindows())
                return "cmd.exe";
            if (OperatingSystem.IsMacOS() || OperatingSystem.IsLinux())
                return "/bin/bash";

            throw new PlatformNotSupportedException("Unsupported OS for shell execution.");
        }

        private static string GetShellArguments(string toolName, string command)
        {
            return OperatingSystem.IsWindows()
                ? $"/C {toolName} {command}"
                : $"-c \"{toolName} {command}\"";
        }
    }
}
