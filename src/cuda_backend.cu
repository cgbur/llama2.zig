#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpsPerBlock = kBlockSize / 32;

char last_error[512] = "no CUDA error";

void set_error(const char* operation, cudaError_t error) {
    std::snprintf(last_error, sizeof(last_error), "%s: %s", operation,
                  cudaGetErrorString(error));
}

struct Config {
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t vocab_size;
    int32_t seq_len;
};

struct Input {
    int32_t token;
    int32_t pos;
    int32_t next;
};

struct State {
    float* x = nullptr;
    float* xb = nullptr;
    float* hb = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* att = nullptr;
    float* logits = nullptr;
    float* key_cache = nullptr;
    float* value_cache = nullptr;
};

struct Weights {
    const __half* token_embedding_table;
    const __half* rms_att_weight;
    const __half* wq;
    const __half* wk;
    const __half* wv;
    const __half* wo;
    const __half* rms_ffn_weight;
    const __half* w1;
    const __half* w2;
    const __half* w3;
    const __half* rms_final_weight;
    const __half* freq_cis_real;
    const __half* freq_cis_imag;
    const __half* wcls;
};

struct Context {
    Config config{};
    __half* weights = nullptr;
    float* staging_weights = nullptr;
    Weights weight_views{};
    State state{};
    Input* input = nullptr;
    Input* host_input = nullptr;
    int32_t* next = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
};

size_t expected_weight_count(const Config& c, bool shared_weights) {
    const size_t dim = c.dim;
    const size_t hidden_dim = c.hidden_dim;
    const size_t layers = c.n_layers;
    const size_t head_size = dim / c.n_heads;
    const size_t kv_dim = dim * c.n_kv_heads / c.n_heads;
    size_t count = 0;
    count += static_cast<size_t>(c.vocab_size) * dim;
    count += layers * dim;
    count += layers * dim * dim;
    count += layers * dim * kv_dim;
    count += layers * dim * kv_dim;
    count += layers * dim * dim;
    count += layers * dim;
    count += layers * dim * hidden_dim;
    count += layers * hidden_dim * dim;
    count += layers * dim * hidden_dim;
    count += dim;
    count += static_cast<size_t>(c.seq_len) * head_size / 2;
    count += static_cast<size_t>(c.seq_len) * head_size / 2;
    if (!shared_weights) count += static_cast<size_t>(c.vocab_size) * dim;
    return count;
}

Weights make_weight_views(const Context* context, bool shared_weights) {
    const Config& c = context->config;
    const size_t dim = c.dim;
    const size_t hidden_dim = c.hidden_dim;
    const size_t layers = c.n_layers;
    const size_t head_size = dim / c.n_heads;
    const size_t kv_dim = dim * c.n_kv_heads / c.n_heads;
    const size_t vocab = c.vocab_size;
    const __half* base = context->weights;
    size_t offset = 0;
    Weights w{};
    w.token_embedding_table = base + offset;
    offset += vocab * dim;
    w.rms_att_weight = base + offset;
    offset += layers * dim;
    w.wq = base + offset;
    offset += layers * dim * dim;
    w.wk = base + offset;
    offset += layers * dim * kv_dim;
    w.wv = base + offset;
    offset += layers * dim * kv_dim;
    w.wo = base + offset;
    offset += layers * dim * dim;
    w.rms_ffn_weight = base + offset;
    offset += layers * dim;
    w.w1 = base + offset;
    offset += layers * dim * hidden_dim;
    w.w2 = base + offset;
    offset += layers * hidden_dim * dim;
    w.w3 = base + offset;
    offset += layers * dim * hidden_dim;
    w.rms_final_weight = base + offset;
    offset += dim;
    w.freq_cis_real = base + offset;
    offset += static_cast<size_t>(c.seq_len) * head_size / 2;
    w.freq_cis_imag = base + offset;
    offset += static_cast<size_t>(c.seq_len) * head_size / 2;
    w.wcls = shared_weights ? w.token_embedding_table : base + offset;
    return w;
}

__device__ __forceinline__ float load_weight(__half value) {
    return __half2float(value);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ float block_sum(float value) {
    __shared__ float warp_values[kWarpsPerBlock];
    __shared__ float result;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    value = threadIdx.x < kWarpsPerBlock ? warp_values[lane] : 0.0f;
    if (warp == 0) value = warp_sum(value);
    if (threadIdx.x == 0) result = value;
    __syncthreads();
    return result;
}

__device__ float block_max(float value) {
    __shared__ float warp_values[kWarpsPerBlock];
    __shared__ float result;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_max(value);
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    value = threadIdx.x < kWarpsPerBlock ? warp_values[lane] : -INFINITY;
    if (warp == 0) value = warp_max(value);
    if (threadIdx.x == 0) result = value;
    __syncthreads();
    return result;
}

__global__ void convert_weights(const float* input, __half* output,
                                size_t count) {
    const size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2half(input[i]);
}

__global__ void embedding(float* output, const __half* weights,
                          const Input* input, int dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) output[i] = load_weight(weights[input->token * dim + i]);
}

__global__ void rmsnorm(float* output, const float* input,
                        const __half* weights, int n) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += input[i] * input[i];
    }
    const float scale = rsqrtf(block_sum(sum) / static_cast<float>(n) + 1e-5f);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i] = input[i] * scale * load_weight(weights[i]);
    }
}

__global__ void matvec(float* output, const float* input,
                       const __half* weights, int rows, int columns) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
    if (row >= rows) return;
    float sum = 0.0f;
    const __half* weight_row = weights + static_cast<size_t>(row) * columns;
    for (int column = lane; column < columns; column += 32) {
        sum = fmaf(input[column], load_weight(weight_row[column]), sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) output[row] = sum;
}

__global__ void matvec_accum(float* output, const float* input,
                             const __half* weights, int rows, int columns) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
    if (row >= rows) return;
    float sum = 0.0f;
    const __half* weight_row = weights + static_cast<size_t>(row) * columns;
    for (int column = lane; column < columns; column += 32) {
        sum = fmaf(input[column], load_weight(weight_row[column]), sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) output[row] += sum;
}

__global__ void matvec2(float* output1, float* output2, const float* input,
                        const __half* weights1, const __half* weights2,
                        int rows, int columns) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
    if (row >= rows) return;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    const size_t row_offset = static_cast<size_t>(row) * columns;
    for (int column = lane; column < columns; column += 32) {
        const float value = input[column];
        sum1 = fmaf(value, load_weight(weights1[row_offset + column]), sum1);
        sum2 = fmaf(value, load_weight(weights2[row_offset + column]), sum2);
    }
    sum1 = warp_sum(sum1);
    sum2 = warp_sum(sum2);
    if (lane == 0) {
        output1[row] = sum1;
        output2[row] = sum2;
    }
}

__global__ void matvec2_rms_swiglu(
    float* output, const float* input, const __half* rms_weights,
    const __half* weights1, const __half* weights2, int rows, int columns) {
    float square_sum = 0.0f;
    for (int i = threadIdx.x; i < columns; i += blockDim.x) {
        square_sum += input[i] * input[i];
    }
    const float scale =
        rsqrtf(block_sum(square_sum) / static_cast<float>(columns) + 1e-5f);
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
    if (row >= rows) return;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    const size_t row_offset = static_cast<size_t>(row) * columns;
    for (int column = lane; column < columns; column += 32) {
        const float value =
            input[column] * scale * load_weight(rms_weights[column]);
        sum1 = fmaf(value, load_weight(weights1[row_offset + column]), sum1);
        sum2 = fmaf(value, load_weight(weights2[row_offset + column]), sum2);
    }
    sum1 = warp_sum(sum1);
    sum2 = warp_sum(sum2);
    if (lane == 0) output[row] = sum1 / (1.0f + expf(-sum1)) * sum2;
}

__global__ void matvec3_rms(
    float* output1, float* output2, float* output3, const float* input,
    const __half* rms_weights, const __half* weights1,
    const __half* weights2, const __half* weights3, int rows, int columns) {
    float square_sum = 0.0f;
    for (int i = threadIdx.x; i < columns; i += blockDim.x) {
        square_sum += input[i] * input[i];
    }
    const float scale =
        rsqrtf(block_sum(square_sum) / static_cast<float>(columns) + 1e-5f);
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
    if (row >= rows) return;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    const size_t row_offset = static_cast<size_t>(row) * columns;
    for (int column = lane; column < columns; column += 32) {
        const float value =
            input[column] * scale * load_weight(rms_weights[column]);
        sum1 = fmaf(value, load_weight(weights1[row_offset + column]), sum1);
        sum2 = fmaf(value, load_weight(weights2[row_offset + column]), sum2);
        sum3 = fmaf(value, load_weight(weights3[row_offset + column]), sum3);
    }
    sum1 = warp_sum(sum1);
    sum2 = warp_sum(sum2);
    sum3 = warp_sum(sum3);
    if (lane == 0) {
        output1[row] = sum1;
        output2[row] = sum2;
        output3[row] = sum3;
    }
}

__global__ void rope_store(float* q, float* k, const float* v,
                           float* key_cache, float* value_cache,
                           const __half* freq_real, const __half* freq_imag,
                           const Input* input, int layer_offset, int dim,
                           int kv_dim, int head_size) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= dim) return;
    const size_t frequency_offset =
        static_cast<size_t>(input->pos) * head_size / 2 +
        (i % head_size) / 2;
    const float cosine = load_weight(freq_real[frequency_offset]);
    const float sine = load_weight(freq_imag[frequency_offset]);
    const float q0 = q[i];
    const float q1 = q[i + 1];
    q[i] = q0 * cosine - q1 * sine;
    q[i + 1] = q0 * sine + q1 * cosine;
    if (i < kv_dim) {
        const float k0 = k[i];
        const float k1 = k[i + 1];
        k[i] = k0 * cosine - k1 * sine;
        k[i + 1] = k0 * sine + k1 * cosine;
        const int cache_offset = layer_offset + input->pos * kv_dim + i;
        key_cache[cache_offset] = k[i];
        key_cache[cache_offset + 1] = k[i + 1];
        value_cache[cache_offset] = v[i];
        value_cache[cache_offset + 1] = v[i + 1];
    }
}

__global__ void attention(float* output, float* scores, const float* query,
                          const float* key_cache, const float* value_cache,
                          const Input* input, int layer_offset, int dim,
                          int n_heads, int n_kv_heads, int seq_len) {
    const int head = blockIdx.x;
    const int head_size = dim / n_heads;
    const int kv_dim = dim * n_kv_heads / n_heads;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int pos = input->pos;

    for (int timestep = warp; timestep <= pos; timestep += kWarpsPerBlock) {
        float score = 0.0f;
        const int key_offset =
            layer_offset + timestep * kv_dim + kv_head * head_size;
        for (int i = lane; i < head_size; i += 32) {
            score = fmaf(query[head * head_size + i], key_cache[key_offset + i],
                         score);
        }
        score = warp_sum(score);
        if (lane == 0) {
            scores[head * seq_len + timestep] =
                score * rsqrtf(static_cast<float>(head_size));
        }
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int timestep = threadIdx.x; timestep <= pos;
         timestep += blockDim.x) {
        local_max = fmaxf(local_max, scores[head * seq_len + timestep]);
    }
    const float maximum = block_max(local_max);
    float local_sum = 0.0f;
    for (int timestep = threadIdx.x; timestep <= pos;
         timestep += blockDim.x) {
        const float value = expf(scores[head * seq_len + timestep] - maximum);
        scores[head * seq_len + timestep] = value;
        local_sum += value;
    }
    const float sum = block_sum(local_sum);
    for (int timestep = threadIdx.x; timestep <= pos;
         timestep += blockDim.x) {
        scores[head * seq_len + timestep] /= sum;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float value = 0.0f;
        for (int timestep = 0; timestep <= pos; ++timestep) {
            const int value_offset =
                layer_offset + timestep * kv_dim + kv_head * head_size + i;
            value = fmaf(scores[head * seq_len + timestep],
                         value_cache[value_offset], value);
        }
        output[head * head_size + i] = value;
    }
}

__device__ __forceinline__ bool better_max(float candidate_value,
                                           int candidate_index,
                                           float current_value,
                                           int current_index) {
    return candidate_value > current_value ||
           (candidate_value == current_value && candidate_index < current_index);
}

__global__ void argmax(const float* values, int n, int32_t* output) {
    float best_value = -INFINITY;
    int best_index = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (better_max(values[i], i, best_value, best_index)) {
            best_value = values[i];
            best_index = i;
        }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        const float other_value =
            __shfl_down_sync(0xffffffff, best_value, offset);
        const int other_index =
            __shfl_down_sync(0xffffffff, best_index, offset);
        if (lane + offset < 32 &&
            better_max(other_value, other_index, best_value, best_index)) {
            best_value = other_value;
            best_index = other_index;
        }
    }
    __shared__ float warp_values[kWarpsPerBlock];
    __shared__ int warp_indices[kWarpsPerBlock];
    if (lane == 0) {
        warp_values[warp] = best_value;
        warp_indices[warp] = best_index;
    }
    __syncthreads();
    if (warp == 0) {
        best_value = lane < kWarpsPerBlock ? warp_values[lane] : -INFINITY;
        best_index = lane < kWarpsPerBlock ? warp_indices[lane] : 0;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            const float other_value =
                __shfl_down_sync(0xffffffff, best_value, offset);
            const int other_index =
                __shfl_down_sync(0xffffffff, best_index, offset);
            if (lane + offset < 32 &&
                better_max(other_value, other_index, best_value, best_index)) {
                best_value = other_value;
                best_index = other_index;
            }
        }
        if (lane == 0) *output = best_index;
    }
}

int vector_blocks(int n) { return (n + kBlockSize - 1) / kBlockSize; }

int matvec_blocks(int rows) {
    return (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
}

void launch_forward(Context* context) {
    const Config& c = context->config;
    const Weights& w = context->weight_views;
    State& s = context->state;
    const int dim = c.dim;
    const int hidden_dim = c.hidden_dim;
    const int kv_dim = dim * c.n_kv_heads / c.n_heads;
    const int head_size = dim / c.n_heads;
    cudaStream_t stream = context->stream;

    embedding<<<vector_blocks(dim), kBlockSize, 0, stream>>>(
        s.x, w.token_embedding_table, context->input, dim);

    for (int layer = 0; layer < c.n_layers; ++layer) {
        if (kv_dim == dim) {
            const size_t offset = static_cast<size_t>(layer) * dim * dim;
            matvec3_rms<<<matvec_blocks(dim), kBlockSize, 0, stream>>>(
                s.q, s.k, s.v, s.x,
                w.rms_att_weight + static_cast<size_t>(layer) * dim,
                w.wq + offset, w.wk + offset, w.wv + offset, dim, dim);
        } else {
            rmsnorm<<<1, kBlockSize, 0, stream>>>(
                s.xb, s.x,
                w.rms_att_weight + static_cast<size_t>(layer) * dim, dim);
            matvec<<<matvec_blocks(dim), kBlockSize, 0, stream>>>(
                s.q, s.xb, w.wq + static_cast<size_t>(layer) * dim * dim, dim,
                dim);
            const size_t offset = static_cast<size_t>(layer) * dim * kv_dim;
            matvec2<<<matvec_blocks(kv_dim), kBlockSize, 0, stream>>>(
                s.k, s.v, s.xb, w.wk + offset, w.wv + offset, kv_dim, dim);
        }

        const int layer_offset = layer * c.seq_len * kv_dim;
        rope_store<<<vector_blocks(dim / 2), kBlockSize, 0, stream>>>(
            s.q, s.k, s.v, s.key_cache, s.value_cache, w.freq_cis_real,
            w.freq_cis_imag, context->input, layer_offset, dim, kv_dim,
            head_size);
        attention<<<c.n_heads, kBlockSize, 0, stream>>>(
            s.xb, s.att, s.q, s.key_cache, s.value_cache, context->input,
            layer_offset, dim, c.n_heads, c.n_kv_heads, c.seq_len);
        matvec_accum<<<matvec_blocks(dim), kBlockSize, 0, stream>>>(
            s.x, s.xb, w.wo + static_cast<size_t>(layer) * dim * dim, dim,
            dim);

        const size_t hidden_offset =
            static_cast<size_t>(layer) * dim * hidden_dim;
        matvec2_rms_swiglu<<<matvec_blocks(hidden_dim), kBlockSize, 0,
                            stream>>>(
            s.hb, s.x,
            w.rms_ffn_weight + static_cast<size_t>(layer) * dim,
            w.w1 + hidden_offset, w.w3 + hidden_offset, hidden_dim, dim);
        matvec_accum<<<matvec_blocks(dim), kBlockSize, 0, stream>>>(
            s.x, s.hb,
            w.w2 + static_cast<size_t>(layer) * hidden_dim * dim, dim,
            hidden_dim);
    }

    rmsnorm<<<1, kBlockSize, 0, stream>>>(s.x, s.x, w.rms_final_weight, dim);
    matvec<<<matvec_blocks(c.vocab_size), kBlockSize, 0, stream>>>(
        s.logits, s.x, w.wcls, c.vocab_size, dim);
}

bool allocate_float(float** pointer, size_t count, const char* name) {
    const cudaError_t error = cudaMalloc(reinterpret_cast<void**>(pointer),
                                         count * sizeof(float));
    if (error != cudaSuccess) {
        set_error(name, error);
        return false;
    }
    return true;
}

void destroy_context(Context* context) {
    if (context == nullptr) return;
    if (context->graph_exec != nullptr) cudaGraphExecDestroy(context->graph_exec);
    if (context->graph != nullptr) cudaGraphDestroy(context->graph);
    if (context->stream != nullptr) cudaStreamDestroy(context->stream);
    if (context->host_input != nullptr) cudaFreeHost(context->host_input);
    cudaFree(context->next);
    cudaFree(context->input);
    cudaFree(context->state.value_cache);
    cudaFree(context->state.key_cache);
    cudaFree(context->state.logits);
    cudaFree(context->state.att);
    cudaFree(context->state.v);
    cudaFree(context->state.k);
    cudaFree(context->state.q);
    cudaFree(context->state.hb);
    cudaFree(context->state.xb);
    cudaFree(context->state.x);
    cudaFree(context->weights);
    cudaFree(context->staging_weights);
    delete context;
}

bool capture_graph(Context* context) {
    cudaError_t error = cudaStreamBeginCapture(context->stream,
                                               cudaStreamCaptureModeGlobal);
    if (error != cudaSuccess) {
        set_error("cudaStreamBeginCapture", error);
        return false;
    }
    cudaMemcpyAsync(context->input, context->host_input, sizeof(Input),
                    cudaMemcpyHostToDevice, context->stream);
    launch_forward(context);
    argmax<<<1, kBlockSize, 0, context->stream>>>(
        context->state.logits, context->config.vocab_size, context->next);
    cudaMemcpyAsync(&context->host_input->next, context->next, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, context->stream);
    error = cudaStreamEndCapture(context->stream, &context->graph);
    if (error != cudaSuccess) {
        set_error("cudaStreamEndCapture", error);
        return false;
    }
    error = cudaGraphInstantiate(&context->graph_exec, context->graph, 0);
    if (error != cudaSuccess) {
        set_error("cudaGraphInstantiate", error);
        return false;
    }
    return true;
}

}  // namespace

extern "C" {

struct Llama2CudaConfig {
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t vocab_size;
    int32_t seq_len;
};

const char* llama2_cuda_last_error() { return last_error; }

void* llama2_cuda_create(const Llama2CudaConfig* public_config,
                         const float* host_weights, size_t weights_count,
                         int32_t shared_weights) {
    if (public_config == nullptr || host_weights == nullptr) {
        std::snprintf(last_error, sizeof(last_error),
                      "invalid CUDA backend arguments");
        return nullptr;
    }
    Context* context = new (std::nothrow) Context();
    if (context == nullptr) {
        std::snprintf(last_error, sizeof(last_error),
                      "unable to allocate CUDA context");
        return nullptr;
    }
    std::memcpy(&context->config, public_config, sizeof(Config));
    const bool weights_are_shared = shared_weights != 0;
    const size_t expected =
        expected_weight_count(context->config, weights_are_shared);
    if (weights_count != expected) {
        std::snprintf(last_error, sizeof(last_error),
                      "checkpoint has %zu weights; CUDA layout expected %zu",
                      weights_count, expected);
        destroy_context(context);
        return nullptr;
    }

#define CUDA_TRY(operation)                                                   \
    do {                                                                      \
        const cudaError_t error = (operation);                                \
        if (error != cudaSuccess) {                                           \
            set_error(#operation, error);                                     \
            destroy_context(context);                                         \
            return nullptr;                                                   \
        }                                                                     \
    } while (false)

    CUDA_TRY(cudaStreamCreateWithFlags(&context->stream,
                                       cudaStreamNonBlocking));
    const size_t source_bytes = weights_count * sizeof(float);
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&context->staging_weights),
                        source_bytes));
    CUDA_TRY(cudaMemcpyAsync(context->staging_weights, host_weights,
                             source_bytes, cudaMemcpyHostToDevice,
                             context->stream));
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&context->weights),
                        weights_count * sizeof(__half)));
    const int conversion_blocks =
        static_cast<int>((weights_count + kBlockSize - 1) / kBlockSize);
    convert_weights<<<conversion_blocks, kBlockSize, 0, context->stream>>>(
        context->staging_weights, context->weights, weights_count);
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaStreamSynchronize(context->stream));
    CUDA_TRY(cudaFree(context->staging_weights));
    context->staging_weights = nullptr;
    context->weight_views = make_weight_views(context, weights_are_shared);

    const Config& c = context->config;
    const size_t kv_dim =
        static_cast<size_t>(c.dim) * c.n_kv_heads / c.n_heads;
    if (!allocate_float(&context->state.x, c.dim, "cudaMalloc x") ||
        !allocate_float(&context->state.xb, c.dim, "cudaMalloc xb") ||
        !allocate_float(&context->state.hb, c.hidden_dim, "cudaMalloc hb") ||
        !allocate_float(&context->state.q, c.dim, "cudaMalloc q") ||
        !allocate_float(&context->state.k, kv_dim, "cudaMalloc k") ||
        !allocate_float(&context->state.v, kv_dim, "cudaMalloc v") ||
        !allocate_float(&context->state.att,
                        static_cast<size_t>(c.n_heads) * c.seq_len,
                        "cudaMalloc attention") ||
        !allocate_float(&context->state.logits, c.vocab_size,
                        "cudaMalloc logits") ||
        !allocate_float(&context->state.key_cache,
                        static_cast<size_t>(c.n_layers) * c.seq_len * kv_dim,
                        "cudaMalloc key cache") ||
        !allocate_float(&context->state.value_cache,
                        static_cast<size_t>(c.n_layers) * c.seq_len * kv_dim,
                        "cudaMalloc value cache")) {
        destroy_context(context);
        return nullptr;
    }
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&context->input),
                        sizeof(Input)));
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&context->next),
                        sizeof(int32_t)));
    CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&context->host_input),
                            sizeof(Input)));
    context->host_input->token = 1;
    context->host_input->pos = 0;
    context->host_input->next = 0;
    CUDA_TRY(cudaStreamSynchronize(context->stream));
    if (!capture_graph(context)) {
        destroy_context(context);
        return nullptr;
    }

#undef CUDA_TRY
    return context;
}

int32_t llama2_cuda_forward(void* opaque_context, int32_t token, int32_t pos,
                            float* host_logits, int32_t* host_next) {
    Context* context = static_cast<Context*>(opaque_context);
    if (context == nullptr || host_next == nullptr || pos < 0 ||
        pos >= context->config.seq_len) {
        std::snprintf(last_error, sizeof(last_error),
                      "invalid CUDA forward arguments");
        return -1;
    }
    context->host_input->token = token;
    context->host_input->pos = pos;
    cudaError_t error = cudaGraphLaunch(context->graph_exec, context->stream);
    if (error != cudaSuccess) {
        set_error("cudaGraphLaunch", error);
        return -1;
    }
    if (host_logits != nullptr) {
        error = cudaMemcpyAsync(host_logits, context->state.logits,
                                static_cast<size_t>(context->config.vocab_size) *
                                    sizeof(float),
                                cudaMemcpyDeviceToHost, context->stream);
        if (error != cudaSuccess) {
            set_error("cudaMemcpyAsync logits", error);
            return -1;
        }
    }
    error = cudaStreamSynchronize(context->stream);
    if (error != cudaSuccess) {
        set_error("cudaStreamSynchronize", error);
        return -1;
    }
    *host_next = context->host_input->next;
    return 0;
}

void llama2_cuda_destroy(void* opaque_context) {
    destroy_context(static_cast<Context*>(opaque_context));
}

}  // extern "C"
