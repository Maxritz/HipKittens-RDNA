# Build system for the gfx1201 (RDNA4) GEMM optimization ladder.
#
# RDNA4 differences from CDNA5:
# - Wave32 threads (WARP_THREADS=32, same as CDNA5)
# - Real WMMA intrinsics (__builtin_amdgcn_wmma_f32_16x16x32_*)
# - IEEE/OCP FP8 support via WMMA
# - Split counters (s_wait_loadcnt, s_wait_dscnt, s_wait_kmcnt, s_wait_vmcnt)
# - 256KB banked LDS
# - No scaled FMMA instructions
#
# RDNA4 supports workgroup clustering via __cluster_dims__, same as CDNA5.

GPU_TARGET ?= RDNA4
CUSTOM_RULES := 1

LADDER := 00_gemm_naive 01_gemm_double_buf 02_gemm_async 03_gemm_128x128 \
          04_gemm_256x256 05_gemm_deepk 06_gemm_segment 07_gemm_tdm \
          08_gemm_split_bar 09_gemm_wgc_multicast 10_gemm_epilogue 11_gemm_one_wave \
          12_gemm_two_waves

KERNEL ?= 00_gemm_naive
SRC    := $(KERNEL).cpp

HK_DEFS ?=

include ../../../../common.mk

PYEXT := $(KERNEL)$(PY_EXT_SUFFIX)

TK_HEADERS := $(shell find $(THUNDERKITTENS_ROOT)/include -name '*.cuh' 2>/dev/null)
LOCAL_HEADERS := harness.h common.h pyext.h

.PHONY: all ladder all-kernels test clean

all: $(PYEXT)

$(PYEXT): $(SRC) $(LOCAL_HEADERS) $(TK_HEADERS)
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) $(PY_INCLUDES) $(PY_LDFLAGS) -shared -fPIC \
		-DHARNESS_PYEXT -DHK_PYEXT_NAME=$(KERNEL) $(HK_DEFS) $(SRC) -o $(PYEXT)

ladder all-kernels:
	@for k in $(LADDER); do $(MAKE) KERNEL=$$k || exit 1; done

test: ladder
	$(PYTHON) test.py

clean:
	rm -f *$(PY_EXT_SUFFIX)
	rm -rf __pycache__