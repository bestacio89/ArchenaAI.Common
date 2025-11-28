using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

namespace ArchenaAI.Common.Semantic.Kernel.Memory.Models
{
    /// <summary>
    /// Represents a vector embedding used for semantic similarity and memory retrieval.
    /// </summary>
    public sealed class Embedding
    {
        private float[] _vector = Array.Empty<float>();

        /// <summary>
        /// Gets or sets the unique identifier for this embedding (optional, usually matches its source record ID).
        /// </summary>
        public string Id { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// Gets or sets the model used to generate this embedding (e.g., "text-embedding-3-large").
        /// </summary>
        public string Model { get; set; } = "text-embedding-3-large";

        /// <summary>
        /// Gets or sets the numeric vector representation of the text.
        /// </summary>
        /// <remarks>
        /// Returns a defensive copy to prevent external mutation of the internal state.
        /// </remarks>
        public IReadOnlyList<float> Vector
        {
            get => _vector;
            set => _vector = value?.ToArray() ?? Array.Empty<float>();
        }

        /// <summary>
        /// Gets optional metadata describing this embedding.
        /// </summary>
        public IDictionary<string, object?> Metadata { get; } = new Dictionary<string, object?>();

        /// <summary>
        /// Gets or sets the timestamp of when this embedding was created.
        /// </summary>
        public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;

        /// <summary>
        /// Gets or sets the text or source content used to generate this embedding (optional).
        /// </summary>
        public string? SourceText { get; set; }

        /// <summary>
        /// Calculates the cosine similarity between two embeddings.
        /// </summary>
        /// <param name="a">The first embedding.</param>
        /// <param name="b">The second embedding.</param>
        /// <returns>A value between -1 and 1 indicating the cosine similarity.</returns>
        /// <exception cref="ArgumentNullException">Thrown when either embedding is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when embeddings have different dimensionalities.</exception>
        public static double CosineSimilarity(Embedding a, Embedding b)
        {
            ArgumentNullException.ThrowIfNull(a);
            ArgumentNullException.ThrowIfNull(b);

            if (a._vector.Length != b._vector.Length)
                throw new ArgumentException("Embeddings must have the same dimensionality.");

            double dot = 0, magA = 0, magB = 0;

            for (int i = 0; i < a._vector.Length; i++)
            {
                dot += a._vector[i] * b._vector[i];
                magA += a._vector[i] * a._vector[i];
                magB += b._vector[i] * b._vector[i];
            }

            var denom = Math.Sqrt(magA) * Math.Sqrt(magB);
            return denom == 0 ? 0 : dot / denom;
        }

        /// <summary>
        /// Normalizes the embedding vector to unit length.
        /// </summary>
        public void Normalize()
        {
            var magnitude = Math.Sqrt(_vector.Sum(v => v * v));
            if (magnitude == 0) return;

            for (int i = 0; i < _vector.Length; i++)
                _vector[i] = (float)(_vector[i] / magnitude);
        }

        /// <summary>
        /// Returns a human-readable representation for debugging.
        /// </summary>
        public override string ToString() =>
            $"{Model} | Dim: {_vector.Length} | Created: {CreatedAt:u}";
    }
}
