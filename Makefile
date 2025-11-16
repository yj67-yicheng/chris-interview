BINARY_NAME = dgemm_x86
CC			= gcc
CFLAGS		= -std=c11 -O0 -march=skylake-avx512 -w -lpthread -fopenmp

# 自动检测 MKL 路径
# 优先使用 Intel 官方安装路径，如果不存在则使用系统路径（apt 安装）
MKL_FOUND = 0
# 检查 Intel 官方安装路径
ifneq ($(wildcard /opt/intel/mkl/include/mkl.h),)
    MKLPATH		= /opt/intel/mkl
    LDFLAGS		= -L$(MKLPATH)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -DMKL_ILP64 -m64
    INCFLAGS	= -I$(MKLPATH)/include
    MKL_FOUND = 1
endif
# 检查 apt 安装路径（/usr/include/mkl/mkl.h）
ifneq ($(wildcard /usr/include/mkl/mkl.h),)
    MKLPATH		= /usr
    LDFLAGS		= -L/usr/lib/x86_64-linux-gnu -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -DMKL_ILP64 -m64
    INCFLAGS	= -I/usr/include/mkl
    MKL_FOUND = 1
endif
# 检查另一个可能的 apt 安装路径（/usr/include/mkl.h）
ifneq ($(wildcard /usr/include/mkl.h),)
    MKLPATH		= /usr
    LDFLAGS		= -L/usr/lib/x86_64-linux-gnu -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -DMKL_ILP64 -m64
    INCFLAGS	= -I/usr/include
    MKL_FOUND = 1
endif

# 如果找不到 MKL，显示错误信息
ifeq ($(MKL_FOUND),0)
    $(error MKL not found! Please install Intel MKL: sudo apt install -y intel-mkl)
endif


SRC			= $(wildcard *.c)
build : $(BINARY_NAME)

$(BINARY_NAME): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS) $(SRC) -o $(BINARY_NAME)

clean:
	rm -f $(BINARY_NAME)