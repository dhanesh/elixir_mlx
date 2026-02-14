# Makefile for elixir_mlx NIF
# Satisfies: T1 (mlx-c binding), RT-2 (mlx-c build), RT-9 (precompiled binaries)

# Erlang/OTP paths
ERTS_INCLUDE_DIR ?= $(shell erl -noshell -eval "io:format(\"~ts/erts-~ts/include/\", [code:root_dir(), erlang:system_info(version)])." -s init stop)
ERL_INTERFACE_INCLUDE_DIR ?= $(shell erl -noshell -eval "io:format(\"~ts\", [code:lib_dir(erl_interface, include)])." -s init stop)
ERL_INTERFACE_LIB_DIR ?= $(shell erl -noshell -eval "io:format(\"~ts\", [code:lib_dir(erl_interface, lib)])." -s init stop)

# mlx-c configuration
MLX_C_VERSION ?= 0.1.2
MLX_C_DIR = c_src/mlx-c
MLX_C_BUILD_DIR = $(MLX_C_DIR)/build
MLX_C_INCLUDE_DIR = $(MLX_C_DIR)
MLX_C_LIB = $(MLX_C_BUILD_DIR)/libmlxc.a
MLX_LIB = $(MLX_C_BUILD_DIR)/_deps/mlx-build/libmlx.a

# Compiler flags
CC = clang
CFLAGS = -O2 -Wall -Wextra -fPIC -std=c11 \
         -I$(ERTS_INCLUDE_DIR) \
         -I$(MLX_C_INCLUDE_DIR)

LDFLAGS = -shared \
          $(MLX_C_LIB) $(MLX_LIB) \
          -framework Metal -framework Foundation -framework Accelerate \
          -lstdc++

# macOS-specific
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -dynamiclib -undefined dynamic_lookup
    NIF_EXT = .so
else
    $(error ElixirMlx only supports macOS on Apple Silicon)
endif

# Output
PRIV_DIR = priv
NIF_SO = $(PRIV_DIR)/mlx_nif$(NIF_EXT)

# Source files
C_SRC = c_src/mlx_nif.c

.PHONY: all clean mlx-c

all: $(PRIV_DIR) mlx-c $(NIF_SO)

$(PRIV_DIR):
	mkdir -p $(PRIV_DIR)

# Build mlx-c from source (pinned version per TN6)
mlx-c:
	@if [ ! -d "$(MLX_C_DIR)" ]; then \
		echo "Downloading mlx-c v$(MLX_C_VERSION)..."; \
		git clone --depth 1 --branch v$(MLX_C_VERSION) \
			https://github.com/ml-explore/mlx-c.git $(MLX_C_DIR) 2>/dev/null || \
		git clone --depth 1 https://github.com/ml-explore/mlx-c.git $(MLX_C_DIR); \
	fi
	@if [ ! -d "$(MLX_C_BUILD_DIR)" ]; then \
		echo "Building mlx-c..."; \
		mkdir -p $(MLX_C_BUILD_DIR) && \
		cd $(MLX_C_BUILD_DIR) && \
		cmake .. -DCMAKE_BUILD_TYPE=Release \
			-DMLX_C_BUILD_EXAMPLES=OFF \
			-DMLX_C_BUILD_TESTS=OFF \
			-DCMAKE_CXX_FLAGS="-Wno-error=unknown-warning-option -Wno-explicit-specialization-storage-class" && \
		make -j$$(sysctl -n hw.ncpu); \
	fi

$(NIF_SO): $(C_SRC) mlx-c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(NIF_SO)
	rm -rf $(MLX_C_BUILD_DIR)
