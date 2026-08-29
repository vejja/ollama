package mlx

import (
	"fmt"
	"testing"
)

func TestGatedDeltaOutputMatchesGraph(t *testing.T) {
	skipIfNoMLX(t)

	var mismatches []string
	withMLXThread(t, func() {
		for _, shape := range [][]int{
			{1, 1, 48, 128},
			{1, 4, 48, 128},
			{2, 11, 48, 128},
			{1, 64, 48, 128},
		} {
			x := patternArray(DTypeFloat16, shape, -0.5, 0.01, 17, 257)
			z := patternArray(DTypeFloat16, shape, -1.0, 0.02, 13, 251)
			weight := patternArray(DTypeFloat16, []int{shape[len(shape)-1]}, 0.5, 0.005, 11, 241)
			ref := gatedDeltaOutputGraph(x, z, weight, 1e-6)
			got := GatedDeltaOutput(x, z, weight, 1e-6)
			if err := requireExact("output", got, ref); err != nil {
				mismatches = append(mismatches, fmt.Sprintf("shape %v: %v", shape, err))
			}
		}
		if gatedDeltaOutput.metalDisabled {
			t.Error("GatedDeltaOutput Metal kernel disabled")
		}
	})
	for _, mismatch := range mismatches {
		t.Error(mismatch)
	}
}
