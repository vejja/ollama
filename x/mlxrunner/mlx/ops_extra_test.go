package mlx

import (
	"math"
	"testing"
)

// fp4Values decodes an fp4 (E2M1) code to its value.
var fp4Values = [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}

func TestDequantizeGlobalScale(t *testing.T) {
	skipIfNoMLX(t)
	withMLXThread(t, func() {
		testDequantizeGlobalScale(t)
	})
}

// The quantized payload is built directly, the way an nvfp4 checkpoint ships
// it: packed fp4 codes, e4m3 group-scale bytes, and a separate global scale.
// Only the dequantize consumer path runs, so expectations are exact.
func testDequantizeGlobalScale(t *testing.T) {
	const rows, cols, group = 4, 64, 16
	// Every group cycles through all 16 codes; group g of row r has scale
	// 2^((r+g)%4-1), a power of two so every expected product is exact.
	scaleOf := func(r, g int) float32 {
		return float32(math.Ldexp(1, (r+g)%4-1))
	}

	packed := make([]uint32, rows*cols/8)
	for i := range packed {
		for j := range 8 {
			packed[i] |= uint32((i*8+j)%16) << (4 * j)
		}
	}
	scaleBits := make([]uint8, rows*(cols/group))
	for i := range scaleBits {
		exp := (i/(cols/group)+i%(cols/group))%4 - 1
		scaleBits[i] = uint8((exp + 7) << 3)
	}
	wq := FromValues(packed, rows, cols/8)
	scales := FromValues(scaleBits, rows, cols/group)

	check := func(name string, got *Array, gs func(r int) float32) {
		t.Helper()
		g32 := got.AsType(DTypeFloat32)
		Eval(g32)
		values := g32.Floats()
		if len(values) != rows*cols {
			t.Errorf("%s: length = %d, want %d", name, len(values), rows*cols)
			return
		}
		for i, v := range values {
			r, c := i/cols, i%cols
			if want := fp4Values[c%16] * scaleOf(r, c/group) * gs(r); v != want {
				t.Errorf("%s[%d] = %v, want %v", name, i, v, want)
				return
			}
		}
	}

	base := Dequantize(wq, scales, nil, group, 4, "nvfp4", nil)
	check("no scale", base, func(int) float32 { return 1 })

	perRow := []float32{0.5, 1, 2, 4}
	cases := []struct {
		name  string
		scale *Array
		gs    func(r int) float32
	}{
		{"scalar", FromValues([]float32{2}, 1), func(int) float32 { return 2 }},
		{"perRow", FromValues(perRow, rows), func(r int) float32 { return perRow[r] }},
	}
	for _, tc := range cases {
		got := Dequantize(wq, scales, nil, group, 4, "nvfp4", tc.scale)
		if got.DType() != base.DType() {
			t.Errorf("%s: dtype = %v, want %v", tc.name, got.DType(), base.DType())
		}
		check(tc.name, got, tc.gs)
	}
}

func TestQuantizedMatmulGlobalScaleBeforeHalfStore(t *testing.T) {
	skipIfNoMLX(t)
	withMLXThread(t, func() {
		const rows, cols, group = 32, 16384, 16
		packed := make([]uint32, rows*cols/8)
		for i := range packed {
			packed[i] = 0x77777777 // Eight E2M1 values of 6.
		}
		scaleBits := make([]uint8, rows*(cols/group))
		for i := range scaleBits {
			scaleBits[i] = 0x38 // E4M3 value 1.
		}

		wq := FromValues(packed, rows, cols/8)
		scales := FromValues(scaleBits, rows, cols/group)
		// MLX convention: effective multiplier times 448 * 6.
		scalarScale := FromValues([]float32{0.2688}, 1)
		rowScaleValues := make([]float32, rows)
		for i := range rowScaleValues {
			rowScaleValues[i] = 0.2688 * float32(i+1)
		}
		rowScales := FromValues(rowScaleValues, rows)

		for _, batch := range []int{1, 32} {
			x := FromValues(make([]float32, batch*cols), batch, cols)
			ones := AddScalar(x, 1).AsType(DTypeFloat16)
			fused := QuantizedMatmul(ones, wq, scales, nil, true, group, 4, "nvfp4", scalarScale).AsType(DTypeFloat32)
			rowFused := QuantizedMatmul(ones, wq, scales, nil, true, group, 4, "nvfp4", rowScales).AsType(DTypeFloat32)
			unscaled := QuantizedMatmul(ones, wq, scales, nil, true, group, 4, "nvfp4", nil).AsType(DTypeFloat32)
			Eval(fused, rowFused, unscaled)

			if got, want := fused.Floats()[0], float32(9.8304); math.Abs(float64(got-want)) > 0.02 {
				t.Fatalf("batch %d fused output = %v, want %v", batch, got, want)
			}
			if got := unscaled.Floats()[0]; !math.IsInf(float64(got), 1) {
				t.Fatalf("batch %d unscaled output = %v, want +Inf", batch, got)
			}
			rowValues := rowFused.Floats()
			for row := range rows {
				if got, want := rowValues[row], float32(9.8304*float32(row+1)); math.Abs(float64(got-want)) > 0.1 {
					t.Fatalf("batch %d row %d fused output = %v, want %v", batch, row, got, want)
				}
			}
		}
	})
}
