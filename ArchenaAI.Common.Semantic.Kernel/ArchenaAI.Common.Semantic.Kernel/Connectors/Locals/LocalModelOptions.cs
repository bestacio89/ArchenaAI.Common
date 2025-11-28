namespace ArchenaAI.Common.Semantic.Kernel.Connectors.Locals
{
    /// <summary>
    /// Represents configuration options for a locally hosted model connector.
    /// This class defines how the Semantic Kernel interacts with on-device
    /// or locally served models such as Ollama, LM Studio, or custom runtimes.
    /// </summary>
    public sealed class LocalModelOptions
    {
        /// <summary>
        /// Gets the absolute or relative file system path to the local model binary or checkpoint.
        /// </summary>
        /// <remarks>
        /// This path is typically used by connectors that load models directly from disk
        /// (e.g., GGUF, ONNX, TorchScript, or other local formats).
        /// </remarks>
        public string ModelPath { get; init; } = string.Empty;

        /// <summary>
        /// Gets the HTTP or WebSocket endpoint of the local model server.
        /// </summary>
        /// <remarks>
        /// Defaults to <c>http://localhost:11434</c>, which corresponds to
        /// the standard Ollama runtime endpoint.  
        /// Can be customized for other local backends (e.g., LM Studio, vLLM, custom Flask server).
        /// </remarks>
        public string Endpoint { get; init; } = "http://localhost:11434";

        /// <summary>
        /// Gets the runtime environment used to execute the model.
        /// </summary>
        /// <remarks>
        /// Common values include:
        /// <list type="bullet">
        /// <item><term><c>python</c></term><description>For Python-based runtimes like PyTorch or Transformers.</description></item>
        /// <item><term><c>cpp</c></term><description>For compiled backends such as llama.cpp or ggml.</description></item>
        /// <item><term><c>rust</c></term><description>For optimized Rust-based inference engines.</description></item>
        /// </list>
        /// </remarks>
        public string Runtime { get; init; } = "python";

        /// <summary>
        /// Gets a value indicating whether GPU acceleration should be used
        /// when running the local model.
        /// </summary>
        /// <remarks>
        /// This flag determines whether CUDA, ROCm, or other hardware backends
        /// are utilized during inference.  
        /// Set to <see langword="false"/> for CPU-only execution.
        /// </remarks>
        public bool UseGPU { get; init; } = true;
    }
}
