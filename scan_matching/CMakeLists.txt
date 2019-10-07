set(SOURCE_FILES
    "common.h"
    "common.cu"
	"cpu.h"
	"cpu.cu"
	"naivegpu.h"
	"naivegpu.cu"
	"svd3.h"
	"svd3_cuda.h"
    )

cuda_add_library(scan_matching
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )
