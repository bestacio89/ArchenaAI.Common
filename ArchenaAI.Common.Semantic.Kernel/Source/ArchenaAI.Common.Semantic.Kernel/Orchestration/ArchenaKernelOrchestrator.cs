using System;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using ArchenaAI.Common.Semantic.Kernel.Skills;
using ArchenaAI.Common.Semantic.Kernel.Memory.Contracts;
using ArchenaAI.Common.Semantic.Kernel.Actions;
using ArchenaAI.Common.Semantic.Kernel.Planning;

namespace ArchenaAI.Common.Semantic.Kernel.Orchestration
{
    /// <summary>
    /// High-level orchestrator implementing reasoning, reflection, correction,
    /// tool-calling, planning, and memory-augmented loops, while broadcasting
    /// semantic events through the distributed SemanticOrchestrator.
    /// </summary>
    public sealed class ArchenaKernelOrchestrator
    {
        private readonly IArchenaKernel _kernel;
        private readonly IMemoryManager _memory;
        private readonly ISkillRegistry _skills;
        private readonly SemanticOrchestrator _semanticBus;
        private readonly LLMActionPlanner _actionPlanner;

        public ArchenaKernelOrchestrator(
            IArchenaKernel kernel,
            IMemoryManager memory,
            ISkillRegistry skills,
            SemanticOrchestrator semanticBus,
            LLMActionPlanner actionPlanner)
        {
            _kernel = kernel;
            _memory = memory;
            _skills = skills;
            _semanticBus = semanticBus;
            _actionPlanner = actionPlanner;
        }

        // ------------------------------------------------------------
        // Main reasoning loop
        // ------------------------------------------------------------
        public async Task<string> RunReasoningLoopAsync(
            string input,
            LoopOptions options,
            CancellationToken ct = default)
        {
            string current = input;
            string? last = null;
            string correlationId = Guid.NewGuid().ToString();

            // Emit initial event
            await _semanticBus.EmitReasoningEventAsync(
                skill: "reasoning.start",
                content: current,
                ct);

            for (int i = 0; i < options.MaxIterations; i++)
            {
                // ------------------------------------------------------------
                // Memory retrieval
                // ------------------------------------------------------------
                if (options.UseMemory)
                {
                    var memories = await _memory.SearchAsync(current, 3, ct);
                    var context = string.Join("\n", memories.Select(m => m.Content));

                    if (!string.IsNullOrWhiteSpace(context))
                        current = $"Context:\n{context}\n\nUser Input:\n{current}";
                }

                // Event: reasoning pass
                await _semanticBus.EmitReasoningEventAsync(
                    skill: "reasoning.pass",
                    content: current,
                    ct);

                // ------------------------------------------------------------
                // Reasoning step
                // ------------------------------------------------------------
                string reasoning = await _kernel.ExecuteSkillAsync(
                    skillName: "reasoning",
                    input: current,
                    ct);

                await _semanticBus.EmitReasoningEventAsync(
                    skill: "reasoning.output",
                    content: reasoning,
                    ct);

                // ------------------------------------------------------------
                // TOOL CALLING PHASE (NEW)
                // ------------------------------------------------------------
                var toolOutput = await _actionPlanner.ProcessAsync(
                    llmResponse: reasoning,
                    correlationId: correlationId,
                    ct: ct);

                if (toolOutput != reasoning)
                {
                    // Tool call executed → override reasoning output
                    reasoning = toolOutput;

                    await _semanticBus.EmitReasoningEventAsync(
                        skill: "action.executed",
                        content: reasoning,
                        ct);
                }

                // ------------------------------------------------------------
                // Reflection step
                // ------------------------------------------------------------
                if (options.UseReflection)
                {
                    string reflection = await _kernel.ExecuteSkillAsync("reflection", reasoning, ct);

                    await _semanticBus.EmitReasoningEventAsync(
                        skill: "reflection",
                        content: reflection,
                        ct);

                    reasoning = reflection;
                }

                // ------------------------------------------------------------
                // Correction step
                // ------------------------------------------------------------
                if (options.UseCorrection)
                {
                    string correction = await _kernel.ExecuteSkillAsync("correction", reasoning, ct);

                    await _semanticBus.EmitReasoningEventAsync(
                        skill: "correction",
                        content: correction,
                        ct);

                    reasoning = correction;
                }

                // ------------------------------------------------------------
                // Termination conditions
                // ------------------------------------------------------------
                if (ShouldTerminate(reasoning, options))
                {
                    await _semanticBus.EmitReasoningEventAsync(
                        skill: "reasoning.finish",
                        content: reasoning,
                        ct);

                    return reasoning;
                }

                if (reasoning == last)
                {
                    await _semanticBus.EmitReasoningEventAsync(
                        skill: "reasoning.stalled",
                        content: reasoning,
                        ct);

                    return reasoning;
                }

                last = reasoning;
                current = reasoning;
            }

            // Max iteration fallback
            await _semanticBus.EmitReasoningEventAsync(
                skill: "reasoning.max_iterations",
                content: last ?? input,
                ct);

            return last ?? input;
        }

        // ------------------------------------------------------------
        // Planning loop (Plan → Execute → Summarize)
        // ------------------------------------------------------------
        public async Task<string> RunPlanExecuteLoopAsync(
            string input,
            LoopOptions options,
            CancellationToken ct = default)
        {
            // Plan
            string plan = await _kernel.ExecuteSkillAsync("planning", input, ct);

            await _semanticBus.EmitReasoningEventAsync("planning", plan, ct);

            // Execute plan steps
            string result = await RunReasoningLoopAsync(plan, options, ct);

            // Summarize
            string summary = await _kernel.ExecuteSkillAsync("summarization", result, ct);

            await _semanticBus.EmitReasoningEventAsync("summarization", summary, ct);

            return summary;
        }

        // ------------------------------------------------------------
        // Termination helper
        // ------------------------------------------------------------
        private bool ShouldTerminate(string output, LoopOptions opt)
        {
            if (string.IsNullOrWhiteSpace(output))
                return true;

            if (opt.TerminationRegex != null &&
                Regex.IsMatch(output, opt.TerminationRegex, RegexOptions.IgnoreCase))
                return true;

            return false;
        }
    }
}
