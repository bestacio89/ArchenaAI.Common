using System;

namespace ArchenaAI.Common.Semantic.Kernel.Observability
{
    /// <summary>
    /// Provides structured logging capabilities for the Semantic Kernel.
    /// Allows introspection of reasoning, skills, and execution results.
    /// </summary>
    public interface IKernelLogger
    {
        /// <summary>
        /// Logs a semantic reasoning step or inner thought.
        /// </summary>
        /// <param name="thought">A string describing the kernel's current reasoning or hypothesis.</param>
        void LogThought(string thought);

        /// <summary>
        /// Logs the invocation of a semantic skill or function.
        /// </summary>
        /// <param name="skill">The name of the skill or function executed.</param>
        /// <param name="result">The resulting output or conclusion from executing the skill.</param>
        void LogSkill(string skill, string result);

        /// <summary>
        /// Logs an error encountered during semantic kernel execution.
        /// </summary>
        /// <param name="ex">The exception representing the error that occurred.</param>
        void LogError(Exception ex);
    }
}
