using System;
using System.Diagnostics;
using System.Runtime.Versioning;
using System.Text;
using Franz.Common.Errors;
using Microsoft.Extensions.Logging;

namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Tools
{
    /// <summary>
    /// Provides high-level control for executing and managing Terraform commands
    /// within the semantic kernel orchestration environment.
    /// </summary>
    [SupportedOSPlatform("windows")]
    [SupportedOSPlatform("linux")]
    [SupportedOSPlatform("osx")]
    public sealed class TerraformExecutor
    {
        private readonly ILogger<TerraformExecutor> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="TerraformExecutor"/> class.
        /// </summary>
        /// <param name="logger">The logger used for diagnostics and telemetry.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="logger"/> is null.</exception>
        public TerraformExecutor(ILogger<TerraformExecutor> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Executes a Terraform command asynchronously in a given working directory.
        /// </summary>
        /// <param name="command">The Terraform subcommand and arguments to execute (for example: "plan -out=tfplan").</param>
        /// <param name="workingDirectory">The directory containing Terraform configuration files (.tf, .tfstate).</param>
        /// <param name="ct">A <see cref="CancellationToken"/> for command cancellation.</param>
        /// <returns>A <see cref="Task{TResult}"/> containing the standard output, error, and exit code.</returns>
        [SupportedOSPlatform("windows")]
        [SupportedOSPlatform("linux")]
        [SupportedOSPlatform("osx")]
        public async Task<(string Output, string Error, int ExitCode)> ExecuteAsync(
            string command,
            string workingDirectory,
            CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(command))
                throw new ArgumentException("Terraform command cannot be null or empty.", nameof(command));

            if (string.IsNullOrWhiteSpace(workingDirectory))
                throw new ArgumentException("Working directory cannot be null or empty.", nameof(workingDirectory));

            var processStartInfo = new ProcessStartInfo
            {
                FileName = "terraform",
                Arguments = command,
                WorkingDirectory = workingDirectory,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            _logger.LogInformation("[Terraform] Executing: terraform {Command} in {Dir}", command, workingDirectory);

            using var process = new Process { StartInfo = processStartInfo, EnableRaisingEvents = true };

            var outputBuilder = new StringBuilder();
            var errorBuilder = new StringBuilder();

            process.OutputDataReceived += (_, e) => { if (e.Data != null) outputBuilder.AppendLine(e.Data); };
            process.ErrorDataReceived += (_, e) => { if (e.Data != null) errorBuilder.AppendLine(e.Data); };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            await process.WaitForExitAsync(ct).ConfigureAwait(false);

            var exitCode = process.ExitCode;
            var output = outputBuilder.ToString().Trim();
            var error = errorBuilder.ToString().Trim();

            if (exitCode != 0)
            {
                _logger.LogWarning("[Terraform] Command failed with exit code {ExitCode}. Error: {Error}", exitCode, error);
            }
            else
            {
                _logger.LogInformation("[Terraform] Command succeeded: {Command}", command);
            }

            return (output, error, exitCode);
        }

        /// <summary>
        /// Verifies that Terraform is installed and accessible in the system PATH.
        /// </summary>
        /// <returns><c>true</c> if Terraform is available; otherwise, <c>false</c>.</returns>
        [SupportedOSPlatform("windows")]
        [SupportedOSPlatform("linux")]
        [SupportedOSPlatform("osx")]
        public bool IsTerraformAvailable()
        {
            try
            {
                using var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "terraform",
                        Arguments = "--version",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                process.WaitForExit(2000);

                return process.ExitCode == 0;
            }
            catch (System.ComponentModel.Win32Exception ex)
            {
                _logger.LogWarning(ex, "[Terraform] Terraform executable not found or inaccessible.");
                return false;
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogWarning(ex, "[Terraform] Invalid operation during Terraform version check.");
                return false;
            }
            catch (Exception ex)
            {
                
                _logger.LogError(ex, "[Terraform] Unexpected error while verifying Terraform installation.");
                throw new TechnicalException (ex.Message);
                
            }
        }
    }
}
