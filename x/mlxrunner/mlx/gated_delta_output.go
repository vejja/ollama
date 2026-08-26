package mlx

// Small token batches are faster as graph operations.
const gatedDeltaOutputMinTokens = 12

const gatedDeltaOutputMetalSource = `
constexpr uint NReads = 4;
constexpr uint SIMDSize = 32;

auto row = threadgroup_position_in_grid.x;
auto lid = thread_position_in_threadgroup.x;
auto simd_lane = thread_index_in_simdgroup;
auto simd_group = simdgroup_index_in_threadgroup;
auto row_offset = row * size_t(C);

threadgroup float local_inv_mean[1];
threadgroup float local_sums[SIMDSize];

float values[NReads];
float acc = 0.0f;
for (uint i = 0; i < NReads; ++i) {
  auto column = lid * NReads + i;
  values[i] = column < C ? static_cast<float>(x[row_offset + column]) : 0.0f;
  acc += values[i] * values[i];
}
acc = simd_sum(acc);

if (simd_group == 0) {
  local_sums[simd_lane] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_lane == 0) {
  local_sums[simd_group] = acc;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simd_group == 0) {
  acc = simd_sum(local_sums[simd_lane]);
  if (simd_lane == 0) {
    local_inv_mean[0] = metal::precise::rsqrt(acc / C + eps);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint i = 0; i < NReads; ++i) {
  auto column = lid * NReads + i;
  if (column < C) {
    auto index = row_offset + column;
    InT normalized = weight[column] *
        static_cast<InT>(values[i] * local_inv_mean[0]);
    float gate = static_cast<float>(z[index]);
    float sigmoid_base = 1.0f / (1.0f + metal::exp(metal::abs(gate)));
    float sigmoid = gate < 0.0f ? sigmoid_base : 1.0f - sigmoid_base;
    volatile float silu = gate * sigmoid;
    y[index] = static_cast<InT>(
        static_cast<float>(normalized) * silu);
  }
}
`

var gatedDeltaOutput = &gpuKernel{
	name:    "gated_delta_output",
	inputs:  []string{"x", "z", "weight", "eps"},
	outputs: []string{"y"},
	metal:   gpuSource{source: gatedDeltaOutputMetalSource},
	fallback: func(launch gpuLaunch) []*Array {
		in := launch.inputs
		return []*Array{gatedDeltaOutputGraph(in[0], in[1], in[2], float32(in[3].Float()))}
	},
}

func gatedDeltaOutputGraph(x, z, weight *Array, eps float32) *Array {
	dtype := x.DType()
	normalized := RMSNormFn(x, weight, eps)
	return Mul(
		normalized.AsType(DTypeFloat32),
		SiLU(z.AsType(DTypeFloat32)),
	).AsType(dtype)
}

// GatedDeltaOutput applies the output RMS norm and FP32 SiLU gate. Large FP16
// inputs use one Metal kernel; other inputs use the ordinary graph.
func GatedDeltaOutput(x, z, weight *Array, eps float32) *Array {
	if x == nil || z == nil || weight == nil || x.DType() != DTypeFloat16 ||
		z.DType() != DTypeFloat16 || weight.DType() != DTypeFloat16 ||
		x.NumDims() != 4 || x.Dim(1) < gatedDeltaOutputMinTokens ||
		!gatedDeltaOutputSameDims(x.Dims(), z.Dims()) ||
		weight.NumDims() != 1 || weight.Dim(0) != x.Dim(x.NumDims()-1) {
		return gatedDeltaOutputGraph(x, z, weight, eps)
	}

	C := x.Dim(x.NumDims() - 1)
	threads := (C + 3) / 4
	threads = (threads + 31) / 32 * 32
	if threads > 1024 {
		return gatedDeltaOutputGraph(x, z, weight, eps)
	}
	rows := x.Size() / C
	outs := gatedDeltaOutput.run(gpuLaunch{
		dtypes: []gpuDTypeArg{{"InT", DTypeFloat16}},
		ints:   []gpuIntArg{{"C", C}},
		outputs: []gpuOutputSpec{
			{"GATED_DELTA_OUTPUT", gatedDeltaOutputDims(x.Dims()), DTypeFloat16},
		},
		grid:        [3]int{rows * threads, 1, 1},
		threadGroup: [3]int{threads, 1, 1},
		inputs:      []*Array{x, z, weight, FromValue(eps)},
	})
	return outs[0]
}

func gatedDeltaOutputSameDims(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func gatedDeltaOutputDims(dims []int) []int32 {
	out := make([]int32, len(dims))
	for i, dim := range dims {
		out[i] = int32(dim)
	}
	return out
}
