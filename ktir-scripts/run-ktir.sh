#!/bin/bash

export TORCH_LOGS=output_code
export TORCH_COMPILE_DEBUG=1
export SPYRE_BACKEND_TARGET=SuperDSC
export TORCH_SPYRE_DEBUG=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=`pwd`/triton-dump
export DXP_DEBUG=1
export SPYRE_INDUCTOR_LOG=1
export SPYRE_INDUCTOR_LOG_LEVEL=DEBUG
# export BUNDLE_HBM_SYMBOLS=1
export UNROLL_LOOPS=0
export LX_PLANNING=0
export SENCORES=1

export TORCH_SPYRE_TRITON=0
export TORCH_SPYRE_KTIR=1
export TORCH_SPYRE_KTIR_CPU=1

export RESULT_DIR="ktir-results"

rm -rf triton-dump
rm -rf torch_compile_debug
rm -rf /tmp/torchinductor_*
rm -rf ~/.triton/cache

PYTHON=${1}

# python3 ${PYTHON}

OUT_DIR=${2}

if [ -z "${OUT_DIR}" ]; then
  echo "Error: second argument (output directory name) is required" >&2
  exit 1
fi

rm -rf ./${RESULT_DIR}/${OUT_DIR}
mkdir -p ./${RESULT_DIR}/${OUT_DIR}

python3 ${PYTHON} > ./${RESULT_DIR}/${OUT_DIR}/${OUT_DIR}.log 2>&1
cp torch_compile_debug/run_*/torchinductor/model__0_inference_0.0/* ./${RESULT_DIR}/${OUT_DIR}/
cp torch_compile_debug/run_*/torchinductor/model__0_inference_0.0/* ./${RESULT_DIR}/${OUT_DIR}/
cp triton-dump/*/* ./${RESULT_DIR}/${OUT_DIR}
cp /tmp/torchinductor_*/inductor-spyre/*/*.ktir ./${RESULT_DIR}/${OUT_DIR}/ 2>/dev/null
