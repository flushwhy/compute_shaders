#include <metal_stdlib>
using namespace metal;

// Fuses ML-phys pair feature assembly into one GPU pass.
// The 12-float state and 28-float output are both float4-aligned.
kernel void build_pair_features(
    const device float4 *states [[buffer(0)]],
    device float4 *output [[buffer(1)]],
    constant uint &bodies [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint pairs_per_batch = bodies * bodies;
    const uint batch = tid / pairs_per_batch;
    const uint pair = tid % pairs_per_batch;
    const uint target = pair / bodies;
    const uint neighbor = pair % bodies;

    const uint state_base = batch * bodies * 3;
    const uint target_base = state_base + target * 3;
    const uint neighbor_base = state_base + neighbor * 3;
    const uint output_base = (batch * pairs_per_batch + pair) * 7;

    const float4 target0 = states[target_base];
    const float4 target1 = states[target_base + 1];
    const float4 target2 = states[target_base + 2];
    const float4 neighbor0 = states[neighbor_base];
    const float4 neighbor1 = states[neighbor_base + 1];
    const float4 neighbor2 = states[neighbor_base + 2];

    output[output_base] = target0;
    output[output_base + 1] = target1;
    output[output_base + 2] = target2;
    output[output_base + 3] = neighbor0;
    output[output_base + 4] = neighbor1;
    output[output_base + 5] = neighbor2;
    output[output_base + 6] = float4(
        neighbor0.x - target0.x,
        neighbor0.y - target0.y,
        neighbor0.w - target0.w,
        neighbor1.x - target1.x
    );
}

// Fuses pair-feature construction and the first pair-encoder linear projection.
// One GPU thread computes one (pair, output-channel) value. The 28 input
// features are assembled in registers, so no pair-feature tensor is written.
kernel void project_pair_features(
    const device float4 *states [[buffer(0)]],
    const device float4 *weights [[buffer(1)]],
    const device float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &bodies [[buffer(4)]],
    constant uint &output_features [[buffer(5)]],
    constant uint &batch_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint pairs_per_batch = bodies * bodies;
    const uint total_pairs = batch_size * pairs_per_batch;
    const uint lane = tid % 32;
    const uint pair_index = tid / 32;
    if (pair_index >= total_pairs) {
        return;
    }

    const uint batch = pair_index / pairs_per_batch;
    const uint pair = pair_index % pairs_per_batch;
    const uint target = pair / bodies;
    const uint neighbor = pair % bodies;

    const uint state_base = batch * bodies * 3;
    const uint target_base = state_base + target * 3;
    const uint neighbor_base = state_base + neighbor * 3;

    const float4 target0 = states[target_base];
    const float4 target1 = states[target_base + 1];
    const float4 target2 = states[target_base + 2];
    const float4 neighbor0 = states[neighbor_base];
    const float4 neighbor1 = states[neighbor_base + 1];
    const float4 neighbor2 = states[neighbor_base + 2];
    const float4 relative = float4(
        neighbor0.x - target0.x,
        neighbor0.y - target0.y,
        neighbor0.w - target0.w,
        neighbor1.x - target1.x
    );

    for (uint channel = lane; channel < output_features; channel += 32) {
        const uint weight_base = channel * 7;
        const device float4 *row = weights + weight_base;
        float value = dot(target0, row[0]);
        value += dot(target1, row[1]);
        value += dot(target2, row[2]);
        value += dot(neighbor0, row[3]);
        value += dot(neighbor1, row[4]);
        value += dot(neighbor2, row[5]);
        value += dot(relative, row[6]);
        output[pair_index * output_features + channel] = value + bias[channel];
    }
}
