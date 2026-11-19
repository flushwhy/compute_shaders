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
