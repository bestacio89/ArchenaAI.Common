namespace ArchenaAI.Common.Semantic.Kernel.Orchestration
{
    public sealed class LoopOptions
    {
        public int MaxIterations { get; set; } = 5;

        public bool UseReflection { get; set; } = true;
        public bool UseCorrection { get; set; } = true;
        public bool UseMemory { get; set; } = true;

        public string? TerminationRegex { get; set; } = @"\b(final answer|completed|done)\b";
    }
}
