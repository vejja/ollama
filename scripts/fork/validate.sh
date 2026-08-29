#!/bin/zsh

set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h:h}"
brew_prefix="${HOMEBREW_PREFIX:-/opt/homebrew}"
python_bin="${PYTHON:-$brew_prefix/bin/python3}"
mlx_source="$repo_root/build/darwin-sources/_deps/mlx-src"
mlx_test_build="$repo_root/build/fork-mlx-tests"
python_venv="$repo_root/build/fork-mlx-python-venv"

[[ -d "$mlx_source" ]] || {
  print -ru2 -- "build the release with scripts/fork/build.sh before validation"
  exit 1
}
[[ -x "$python_bin" ]] || {
  print -ru2 -- "Python is required for targeted MLX numerical tests: $python_bin"
  exit 1
}

cd "$repo_root"
go test ./envconfig ./x/mlxrunner/... ./x/models/nn ./x/models/qwen3_5

rm -rf "$mlx_test_build"
cmake -S "$mlx_source" -B "$mlx_test_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_TESTS=ON \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF
cmake --build "$mlx_test_build" --target tests --parallel "${OLLAMA_BUILD_PARALLEL:-$(getconf _NPROCESSORS_ONLN)}"
ctest --test-dir "$mlx_test_build" --output-on-failure

cd "$mlx_source"
for path in build/temp.*(N) build/lib.*(N); do
  rm -rf -- "$path"
done
rm -rf "$python_venv"
"$python_bin" -m venv "$python_venv"
python_runner="$python_venv/bin/python"
"$python_runner" -m pip install --disable-pip-version-check setuptools wheel
"$python_runner" setup.py build_ext --inplace
PYTHONPATH="$mlx_source/python:$mlx_source/python/tests" \
  "$python_runner" -m unittest \
  python.tests.test_quantized.TestQuantized.test_qmm_global_scale_error_cases \
  python.tests.test_quantized.TestQuantized.test_nvfp4_qmm_global_scale \
  python.tests.test_quantized.TestQuantized.test_nvfp4_qmm_per_output_global_scale \
  python.tests.test_quantized.TestQuantized.test_nvfp4_qmv_wide_per_output_global_scale \
  python.tests.test_quantized.TestQuantized.test_nvfp4_qmm_large_m_tile_global_scale \
  python.tests.test_quantized.TestQuantized.test_nvfp4_qmm_global_scale_grad
