set(SOURCE_FILES
    "common.h"
    "common.cu"
    "icp.h"
    "icp.cu"
    )

cuda_add_library(scan_matching
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )
