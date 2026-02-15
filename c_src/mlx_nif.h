// mlx_nif.h - NIF resource type definitions for elixir_mlx
// Satisfies: RT-3 (every mlx-c object has a NIF resource with destructor)
// Satisfies: S2 (no memory leaks from NIF resources)

#ifndef MLX_NIF_H
#define MLX_NIF_H

#include <erl_nif.h>
#include <mlx/c/mlx.h>
#include <mlx/c/optional.h>

// --- Resource Types ---
// Each mlx-c object type gets a corresponding Erlang resource type
// with a destructor that calls the appropriate mlx_*_free function.
// This ensures GC-safe cleanup when Elixir references are collected.

extern ErlNifResourceType *MLX_ARRAY_RESOURCE;
extern ErlNifResourceType *MLX_STREAM_RESOURCE;
extern ErlNifResourceType *MLX_DEVICE_RESOURCE;
extern ErlNifResourceType *MLX_VECTOR_ARRAY_RESOURCE;

// --- Resource Wrappers ---
// We wrap mlx-c handles in structs stored as NIF resources.
// The destructor is called automatically when the Erlang term is GC'd.

typedef struct {
    mlx_array inner;
} MlxArrayResource;

typedef struct {
    mlx_stream inner;
} MlxStreamResource;

typedef struct {
    mlx_device inner;
} MlxDeviceResource;

typedef struct {
    mlx_vector_array inner;
} MlxVectorArrayResource;

// --- Atoms ---
extern ERL_NIF_TERM ATOM_OK;
extern ERL_NIF_TERM ATOM_ERROR;
extern ERL_NIF_TERM ATOM_TRUE;
extern ERL_NIF_TERM ATOM_FALSE;
extern ERL_NIF_TERM ATOM_NIL;

// Dtype atoms
extern ERL_NIF_TERM ATOM_BOOL;
extern ERL_NIF_TERM ATOM_U8;
extern ERL_NIF_TERM ATOM_U16;
extern ERL_NIF_TERM ATOM_U32;
extern ERL_NIF_TERM ATOM_U64;
extern ERL_NIF_TERM ATOM_S8;
extern ERL_NIF_TERM ATOM_S16;
extern ERL_NIF_TERM ATOM_S32;
extern ERL_NIF_TERM ATOM_S64;
extern ERL_NIF_TERM ATOM_F16;
extern ERL_NIF_TERM ATOM_F32;
extern ERL_NIF_TERM ATOM_BF16;
extern ERL_NIF_TERM ATOM_C64;

// Device atoms
extern ERL_NIF_TERM ATOM_CPU;
extern ERL_NIF_TERM ATOM_GPU;

// --- Helper macros ---
#define MLX_NIF_OK(env, term) enif_make_tuple2(env, ATOM_OK, term)
#define MLX_NIF_ERROR(env, msg) enif_make_tuple2(env, ATOM_ERROR, enif_make_string(env, msg, ERL_NIF_LATIN1))

#endif // MLX_NIF_H
