#include <metal_stdlib>
using namespace metal;

// One thread writes one (batch, output_frame, hidden_channel) value.
kernel void length_regulate(
    device const float *states [[buffer(0)]],
    device const int *exclusive_cumsum [[buffer(1)]],
    device const int *mel_lengths [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant int &source_length [[buffer(4)]],
    constant int &hidden_size [[buffer(5)]],
    constant int &output_length [[buffer(6)]],
    constant int &batch_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    const int values_per_batch = output_length * hidden_size;
    const int batch = (int)tid / values_per_batch;
    const int within_batch = (int)tid % values_per_batch;
    const int frame = within_batch / hidden_size;
    const int hidden = within_batch % hidden_size;

    if (batch >= batch_size) {
        return;
    }

    const int output_index = (int)tid;
    if (frame >= mel_lengths[batch]) {
        output[output_index] = 0.0f;
        return;
    }

    const device int *prefix = exclusive_cumsum + batch * source_length;
    int low = 0;
    int high = source_length - 1;
    int source = 0;

    // Find the last phoneme whose exclusive prefix is <= this output frame.
    // Choosing the last equal prefix also handles zero-duration phonemes.
    while (low <= high) {
        const int middle = (low + high) / 2;
        if (prefix[middle] <= frame) {
            source = middle;
            low = middle + 1;
        } else {
            high = middle - 1;
        }
    }

    output[output_index] =
        states[(batch * source_length + source) * hidden_size + hidden];
}

// One thread accumulates the gradient for one input (batch, phoneme, channel).
kernel void length_regulate_backward(
    device const float *grad_output [[buffer(0)]],
    device const int *exclusive_cumsum [[buffer(1)]],
    device const int *mel_lengths [[buffer(2)]],
    device float *grad_states [[buffer(3)]],
    constant int &source_length [[buffer(4)]],
    constant int &hidden_size [[buffer(5)]],
    constant int &output_length [[buffer(6)]],
    constant int &batch_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    const int values_per_batch = source_length * hidden_size;
    const int batch = (int)tid / values_per_batch;
    const int within_batch = (int)tid % values_per_batch;
    const int source = within_batch / hidden_size;
    const int hidden = within_batch % hidden_size;

    if (batch >= batch_size) {
        return;
    }

    const int prefix_index = batch * source_length + source;
    const int start = exclusive_cumsum[prefix_index];
    const int end = source + 1 < source_length
        ? exclusive_cumsum[prefix_index + 1]
        : mel_lengths[batch];

    float gradient = 0.0f;
    for (int frame = start; frame < end && frame < output_length; ++frame) {
        gradient += grad_output[
            (batch * output_length + frame) * hidden_size + hidden
        ];
    }

    grad_states[tid] = gradient;
}
