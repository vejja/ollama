package mlxrunner

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestAdaptNVFP4ComputeDType(t *testing.T) {
	thread, err := mlxthread.Start("dtype-adapter-test", func() error {
		if err := mlx.CheckInit(); err != nil {
			return err
		}
		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
		}
		return nil
	})
	if err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	defer func() {
		if err := thread.Stop(context.Background(), func() {
			mlx.Sweep()
			mlx.ClearCache()
		}); err != nil {
			t.Fatal(err)
		}
	}()

	if err := thread.Do(context.Background(), func() error {
		tensors := map[string]*mlx.Array{
			"support": mlx.FromValues([]float32{1.25, -2.5}, 2).AsType(mlx.DTypeBFloat16),
			"scale":   mlx.FromValues([]float32{3.5}, 1),
			"packed":  mlx.FromValues([]uint32{0x12345678}, 1),
		}
		if err := adaptNVFP4ComputeDType(tensors, "NVFP4", "fp16"); err != nil {
			return err
		}
		if got := tensors["support"].DType(); got != mlx.DTypeFloat16 {
			t.Fatalf("support dtype = %v, want %v", got, mlx.DTypeFloat16)
		}
		if got := tensors["scale"].DType(); got != mlx.DTypeFloat32 {
			t.Fatalf("scale dtype = %v, want %v", got, mlx.DTypeFloat32)
		}
		if got := tensors["packed"].DType(); got != mlx.DTypeUint32 {
			t.Fatalf("packed dtype = %v, want %v", got, mlx.DTypeUint32)
		}
		supportValues := tensors["support"].AsType(mlx.DTypeFloat32)
		mlx.Eval(supportValues)
		if got := supportValues.Floats(); len(got) != 2 || got[0] != 1.25 || got[1] != -2.5 {
			t.Fatalf("support values = %v, want [1.25 -2.5]", got)
		}

		if err := adaptNVFP4ComputeDType(tensors, "NVFP4", "bfloat16"); err != nil {
			return err
		}
		if got := tensors["support"].DType(); got != mlx.DTypeBFloat16 {
			t.Fatalf("support dtype = %v, want %v", got, mlx.DTypeBFloat16)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestAdaptNVFP4ComputeDTypeValidation(t *testing.T) {
	for _, tc := range []struct {
		name      string
		quantType string
		requested string
		wantError string
	}{
		{name: "default", quantType: "NVFP4", requested: ""},
		{name: "auto", quantType: "NVFP4", requested: "auto"},
		{name: "other quantization is unchanged", quantType: "MXFP4", requested: "float16"},
		{name: "invalid dtype", quantType: "NVFP4", requested: "float32", wantError: "expected auto, float16, or bfloat16"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := adaptNVFP4ComputeDType(nil, tc.quantType, tc.requested)
			if tc.wantError == "" && err != nil {
				t.Fatal(err)
			}
			if tc.wantError != "" && (err == nil || !strings.Contains(err.Error(), tc.wantError)) {
				t.Fatalf("error = %v, want substring %q", err, tc.wantError)
			}
		})
	}
}
