use criterion::{black_box, criterion_group, criterion_main, Criterion};
use model2vec_rust::Model2Vec;

fn criterion_benchmark(c: &mut Criterion) {
    // Load the model once
    let model = Model2Vec::load("model").expect("Failed to load Model2Vec model");

    let single_sentence = "This is a single short sentence used for latency performance testing.";
    
    // Benchmark single sentence encoding latency
    c.bench_function("encode_single_sentence", |b| {
        b.iter(|| {
            let _ = model.encode(black_box(single_sentence), black_box(true));
        })
    });

    // Create a batch of sentences for throughput testing
    let batch_sentences = vec![
        "Hello world",
        "Rust is awesome",
        "This is an extremely performant and simple model2vec implementation.",
        "Model2Vec distilled models are up to 500x faster than sentence transformers on CPU.",
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        "Very long sentence containing many words to verify how the pooling scales with token count in Rust.",
        "Another sentence to make a diverse batch of sentences.",
        "Short sentence.",
        "Static embeddings allow extremely fast computation on edge devices without GPU support.",
        "Let's test the scaling of multi-threaded encoding using rayon parallel iterator."
    ];

    // Benchmark batch encoding (Vec<Vec<f32>>)
    c.bench_function("encode_batch_10_sentences", |b| {
        b.iter(|| {
            let _ = model.encode_batch(black_box(&batch_sentences), black_box(true));
        })
    });

    // Benchmark flat batch encoding (Vec<f32>) - which is more performant as it avoids sub-vector allocations
    c.bench_function("encode_batch_flat_10_sentences", |b| {
        b.iter(|| {
            let _ = model.encode_batch_flat(black_box(&batch_sentences), black_box(true));
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
