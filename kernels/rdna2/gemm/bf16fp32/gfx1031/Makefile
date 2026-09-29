# Build system for the gfx1031 (RDNA2) GEMM optimization ladder.
#
# RDNA2 differences from CDNA5:
# - Wave64 threads (WARP_THREADS=64, not 32)
# - No MFMA; uses v_pk_fmac_f32 packed fma chains
# - No FP8 support (static_assert'd in kittens)
# - No workgroup clustering (no __cluster_dims__)
# - 64KB LDS (not 256KB banked)
# - Unified s_waitcnt (no split counters)
#
# The cluster rungs (09-12) are adapted to use workgroup ID swizzling
# instead of __cluster_dims__.

GPU_TARGET ?= RDNA2
 CUSTOM_RULES := 1

LADDER := 00_gemm_naive 01_gemm_double_buf 02_gemm_async 03_gemm_128x128 \
          04_gemm_256x256 05_gemm_deepk 06_gemm_segment 07_gemm_tdm \
          08_gemm_split_bar 09_gemm_wgc 10_gemm_epilogue 11_gemm_one_wave \
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