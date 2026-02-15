// mlx_nif.c - Erlang NIF bindings for mlx-c
// Satisfies: T1 (NIF via mlx-c), RT-3 (resource with destructor), S1 (no leaks), S2 (no crashes)

#include "mlx_nif.h"
#include <string.h>
#include <stdlib.h>

// --- Thread-local error buffer (RT-4: error handler) ---
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define THREAD_LOCAL _Thread_local
#elif defined(__GNUC__) || defined(__clang__)
#define THREAD_LOCAL __thread
#else
#define THREAD_LOCAL
#endif

static THREAD_LOCAL char last_error_msg[1024] = "unknown error";

static void mlx_error_handler(const char* msg, void* data) {
    (void)data;
    strncpy(last_error_msg, msg, sizeof(last_error_msg) - 1);
    last_error_msg[sizeof(last_error_msg) - 1] = '\0';
}

// --- Resource type globals ---
ErlNifResourceType *MLX_ARRAY_RESOURCE  = NULL;
ErlNifResourceType *MLX_STREAM_RESOURCE = NULL;
ErlNifResourceType *MLX_DEVICE_RESOURCE = NULL;
ErlNifResourceType *MLX_VECTOR_ARRAY_RESOURCE = NULL;

// --- Atom globals ---
ERL_NIF_TERM ATOM_OK;
ERL_NIF_TERM ATOM_ERROR;
ERL_NIF_TERM ATOM_TRUE;
ERL_NIF_TERM ATOM_FALSE;
ERL_NIF_TERM ATOM_NIL;

ERL_NIF_TERM ATOM_BOOL;
ERL_NIF_TERM ATOM_U8;
ERL_NIF_TERM ATOM_U16;
ERL_NIF_TERM ATOM_U32;
ERL_NIF_TERM ATOM_U64;
ERL_NIF_TERM ATOM_S8;
ERL_NIF_TERM ATOM_S16;
ERL_NIF_TERM ATOM_S32;
ERL_NIF_TERM ATOM_S64;
ERL_NIF_TERM ATOM_F16;
ERL_NIF_TERM ATOM_F32;
ERL_NIF_TERM ATOM_BF16;
ERL_NIF_TERM ATOM_C64;

ERL_NIF_TERM ATOM_CPU;
ERL_NIF_TERM ATOM_GPU;

// --- Resource destructors (S1: GC-safe cleanup) ---
static void array_destructor(ErlNifEnv* env, void* obj) {
    (void)env;
    MlxArrayResource *res = (MlxArrayResource*)obj;
    if (res->inner.ctx) {
        mlx_array_free(res->inner);
        res->inner.ctx = NULL;
    }
}

static void stream_destructor(ErlNifEnv* env, void* obj) {
    (void)env;
    MlxStreamResource *res = (MlxStreamResource*)obj;
    if (res->inner.ctx) {
        mlx_stream_free(res->inner);
        res->inner.ctx = NULL;
    }
}

static void device_destructor(ErlNifEnv* env, void* obj) {
    (void)env;
    MlxDeviceResource *res = (MlxDeviceResource*)obj;
    if (res->inner.ctx) {
        mlx_device_free(res->inner);
        res->inner.ctx = NULL;
    }
}

static void vector_array_destructor(ErlNifEnv* env, void* obj) {
    (void)env;
    MlxVectorArrayResource *res = (MlxVectorArrayResource*)obj;
    if (res->inner.ctx) {
        mlx_vector_array_free(res->inner);
        res->inner.ctx = NULL;
    }
}

// --- Helper: wrap mlx_array in NIF resource ---
// wrap_array_raw returns the resource term without {:ok, ...} wrapper
static ERL_NIF_TERM wrap_array_raw(ErlNifEnv* env, mlx_array arr) {
    MlxArrayResource *res = enif_alloc_resource(MLX_ARRAY_RESOURCE, sizeof(MlxArrayResource));
    res->inner = arr;
    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return term;
}

static ERL_NIF_TERM wrap_array(ErlNifEnv* env, mlx_array arr) {
    return MLX_NIF_OK(env, wrap_array_raw(env, arr));
}

static ERL_NIF_TERM wrap_stream(ErlNifEnv* env, mlx_stream s) {
    MlxStreamResource *res = enif_alloc_resource(MLX_STREAM_RESOURCE, sizeof(MlxStreamResource));
    res->inner = s;
    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return MLX_NIF_OK(env, term);
}

static ERL_NIF_TERM wrap_device(ErlNifEnv* env, mlx_device d) {
    MlxDeviceResource *res = enif_alloc_resource(MLX_DEVICE_RESOURCE, sizeof(MlxDeviceResource));
    res->inner = d;
    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return MLX_NIF_OK(env, term);
}

// --- Helper: atom <-> mlx_dtype mapping (RT-7) ---
static int dtype_from_atom(ErlNifEnv* env, ERL_NIF_TERM atom, mlx_dtype* out) {
    if (enif_compare(atom, ATOM_BOOL) == 0) { *out = MLX_BOOL; return 1; }
    if (enif_compare(atom, ATOM_U8)   == 0) { *out = MLX_UINT8; return 1; }
    if (enif_compare(atom, ATOM_U16)  == 0) { *out = MLX_UINT16; return 1; }
    if (enif_compare(atom, ATOM_U32)  == 0) { *out = MLX_UINT32; return 1; }
    if (enif_compare(atom, ATOM_U64)  == 0) { *out = MLX_UINT64; return 1; }
    if (enif_compare(atom, ATOM_S8)   == 0) { *out = MLX_INT8; return 1; }
    if (enif_compare(atom, ATOM_S16)  == 0) { *out = MLX_INT16; return 1; }
    if (enif_compare(atom, ATOM_S32)  == 0) { *out = MLX_INT32; return 1; }
    if (enif_compare(atom, ATOM_S64)  == 0) { *out = MLX_INT64; return 1; }
    if (enif_compare(atom, ATOM_F16)  == 0) { *out = MLX_FLOAT16; return 1; }
    if (enif_compare(atom, ATOM_F32)  == 0) { *out = MLX_FLOAT32; return 1; }
    if (enif_compare(atom, ATOM_BF16) == 0) { *out = MLX_BFLOAT16; return 1; }
    if (enif_compare(atom, ATOM_C64)  == 0) { *out = MLX_COMPLEX64; return 1; }
    return 0;
}

static ERL_NIF_TERM dtype_to_atom(ErlNifEnv* env, mlx_dtype dt) {
    (void)env;
    switch (dt) {
        case MLX_BOOL:      return ATOM_BOOL;
        case MLX_UINT8:     return ATOM_U8;
        case MLX_UINT16:    return ATOM_U16;
        case MLX_UINT32:    return ATOM_U32;
        case MLX_UINT64:    return ATOM_U64;
        case MLX_INT8:      return ATOM_S8;
        case MLX_INT16:     return ATOM_S16;
        case MLX_INT32:     return ATOM_S32;
        case MLX_INT64:     return ATOM_S64;
        case MLX_FLOAT16:   return ATOM_F16;
        case MLX_FLOAT32:   return ATOM_F32;
        case MLX_BFLOAT16:  return ATOM_BF16;
        case MLX_COMPLEX64: return ATOM_C64;
        default:            return ATOM_F32;
    }
}

// --- Helper: extract shape list -> int array ---
static int get_int_list(ErlNifEnv* env, ERL_NIF_TERM list, int* out, int* len) {
    unsigned int ulen;
    if (!enif_get_list_length(env, list, &ulen)) return 0;
    *len = (int)ulen;
    ERL_NIF_TERM head, tail = list;
    for (unsigned int i = 0; i < ulen; i++) {
        if (!enif_get_list_cell(env, tail, &head, &tail)) return 0;
        if (!enif_get_int(env, head, &out[i])) return 0;
    }
    return 1;
}

// ============================================================
// NIF: Array Creation
// ============================================================

// from_binary(binary, shape_list, dtype_atom) -> {:ok, ref}
static ERL_NIF_TERM nif_from_binary(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    ErlNifBinary bin;
    if (!enif_inspect_binary(env, argv[0], &bin))
        return MLX_NIF_ERROR(env, "expected binary");

    int shape[16], ndim;
    if (!get_int_list(env, argv[1], shape, &ndim))
        return MLX_NIF_ERROR(env, "expected shape list");

    mlx_dtype dtype;
    if (!dtype_from_atom(env, argv[2], &dtype))
        return MLX_NIF_ERROR(env, "invalid dtype");

    mlx_array arr = mlx_array_new_data(bin.data, shape, ndim, dtype);
    if (!arr.ctx) return MLX_NIF_ERROR(env, "failed to create array");
    return wrap_array(env, arr);
}

// zeros(shape_list, dtype_atom, stream) -> {:ok, ref}
static ERL_NIF_TERM nif_zeros(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    int shape[16], ndim;
    if (!get_int_list(env, argv[0], shape, &ndim))
        return MLX_NIF_ERROR(env, "expected shape list");

    mlx_dtype dtype;
    if (!dtype_from_atom(env, argv[1], &dtype))
        return MLX_NIF_ERROR(env, "invalid dtype");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_zeros(&result, shape, ndim, dtype, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ones(shape_list, dtype_atom, stream) -> {:ok, ref}
static ERL_NIF_TERM nif_ones(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    int shape[16], ndim;
    if (!get_int_list(env, argv[0], shape, &ndim))
        return MLX_NIF_ERROR(env, "expected shape list");

    mlx_dtype dtype;
    if (!dtype_from_atom(env, argv[1], &dtype))
        return MLX_NIF_ERROR(env, "invalid dtype");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_ones(&result, shape, ndim, dtype, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// full(shape_list, val_array, dtype_atom, stream) -> {:ok, ref}
static ERL_NIF_TERM nif_full(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    int shape[16], ndim;
    if (!get_int_list(env, argv[0], shape, &ndim))
        return MLX_NIF_ERROR(env, "expected shape list");

    MlxArrayResource *val;
    if (!enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&val))
        return MLX_NIF_ERROR(env, "expected array for fill value");

    mlx_dtype dtype;
    if (!dtype_from_atom(env, argv[2], &dtype))
        return MLX_NIF_ERROR(env, "invalid dtype");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_full(&result, shape, ndim, val->inner, dtype, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// eye(n, m, k, dtype_atom, stream) -> {:ok, ref}
static ERL_NIF_TERM nif_eye(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    int n, m, k;
    if (!enif_get_int(env, argv[0], &n) ||
        !enif_get_int(env, argv[1], &m) ||
        !enif_get_int(env, argv[2], &k))
        return MLX_NIF_ERROR(env, "expected integers for n, m, k");

    mlx_dtype dtype;
    if (!dtype_from_atom(env, argv[3], &dtype))
        return MLX_NIF_ERROR(env, "invalid dtype");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_eye(&result, n, m, k, dtype, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// arange(start, stop, step, dtype_atom, stream) -> {:ok, ref}
static ERL_NIF_TERM nif_arange(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    double start, stop, step;
    if (!enif_get_double(env, argv[0], &start) ||
        !enif_get_double(env, argv[1], &stop) ||
        !enif_get_double(env, argv[2], &step))
        return MLX_NIF_ERROR(env, "expected doubles for start, stop, step");

    mlx_dtype dtype;
    if (!dtype_from_atom(env, argv[3], &dtype))
        return MLX_NIF_ERROR(env, "invalid dtype");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_arange(&result, start, stop, step, dtype, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// linspace(start, stop, num, dtype_atom, stream) -> {:ok, ref}
static ERL_NIF_TERM nif_linspace(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    double start, stop;
    int num;
    if (!enif_get_double(env, argv[0], &start) ||
        !enif_get_double(env, argv[1], &stop) ||
        !enif_get_int(env, argv[2], &num))
        return MLX_NIF_ERROR(env, "expected doubles for start/stop, int for num");

    mlx_dtype dtype;
    if (!dtype_from_atom(env, argv[3], &dtype))
        return MLX_NIF_ERROR(env, "invalid dtype");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_linspace(&result, start, stop, num, dtype, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ============================================================
// NIF: Array Properties
// ============================================================

// shape(arr) -> {:ok, [int]}
static ERL_NIF_TERM nif_shape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    size_t ndim = mlx_array_ndim(a->inner);
    const int* dims = mlx_array_shape(a->inner);

    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int i = (int)ndim - 1; i >= 0; i--) {
        list = enif_make_list_cell(env, enif_make_int(env, dims[i]), list);
    }
    return MLX_NIF_OK(env, list);
}

// dtype(arr) -> {:ok, atom}
static ERL_NIF_TERM nif_dtype(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    mlx_dtype dt = mlx_array_dtype(a->inner);
    return MLX_NIF_OK(env, dtype_to_atom(env, dt));
}

// ndim(arr) -> {:ok, int}
static ERL_NIF_TERM nif_ndim(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");
    return MLX_NIF_OK(env, enif_make_int(env, (int)mlx_array_ndim(a->inner)));
}

// size(arr) -> {:ok, int}
static ERL_NIF_TERM nif_size(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");
    return MLX_NIF_OK(env, enif_make_uint64(env, mlx_array_size(a->inner)));
}

// nbytes(arr) -> {:ok, int}
static ERL_NIF_TERM nif_nbytes(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");
    return MLX_NIF_OK(env, enif_make_uint64(env, mlx_array_nbytes(a->inner)));
}

// ============================================================
// NIF: Array Data Access
// ============================================================

// eval(arr) -> :ok  (dirty scheduler)
static ERL_NIF_TERM nif_eval(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    int ret = mlx_array_eval(a->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return ATOM_OK;
}

// to_binary(arr) -> {:ok, binary}  (dirty scheduler - evals first)
static ERL_NIF_TERM nif_to_binary(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    // Evaluate to materialize lazy computation
    int ret = mlx_array_eval(a->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);

    // Force contiguous layout — broadcast/transpose/slice produce views
    // whose data pointer refers to the original buffer, not the full output.
    mlx_array contiguous = mlx_array_new();
    mlx_stream default_s = mlx_default_gpu_stream_new();
    int c_ret = mlx_contiguous(&contiguous, a->inner, false, default_s);
    mlx_stream_free(default_s);
    if (c_ret != 0) {
        mlx_array_free(contiguous);
        return MLX_NIF_ERROR(env, last_error_msg);
    }
    int e_ret = mlx_array_eval(contiguous);
    if (e_ret != 0) {
        mlx_array_free(contiguous);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    size_t nbytes = mlx_array_nbytes(contiguous);
    mlx_dtype dt = mlx_array_dtype(contiguous);

    const void* data = NULL;
    switch (dt) {
        case MLX_BOOL:      data = mlx_array_data_bool(contiguous); break;
        case MLX_UINT8:     data = mlx_array_data_uint8(contiguous); break;
        case MLX_UINT16:    data = mlx_array_data_uint16(contiguous); break;
        case MLX_UINT32:    data = mlx_array_data_uint32(contiguous); break;
        case MLX_UINT64:    data = mlx_array_data_uint64(contiguous); break;
        case MLX_INT8:      data = mlx_array_data_int8(contiguous); break;
        case MLX_INT16:     data = mlx_array_data_int16(contiguous); break;
        case MLX_INT32:     data = mlx_array_data_int32(contiguous); break;
        case MLX_INT64:     data = mlx_array_data_int64(contiguous); break;
        case MLX_FLOAT16:   data = mlx_array_data_float16(contiguous); break;
        case MLX_FLOAT32:   data = mlx_array_data_float32(contiguous); break;
        case MLX_BFLOAT16:  data = mlx_array_data_bfloat16(contiguous); break;
        case MLX_COMPLEX64: data = mlx_array_data_complex64(contiguous); break;
        default:
            mlx_array_free(contiguous);
            return MLX_NIF_ERROR(env, "unsupported dtype for data access");
    }

    if (!data) {
        mlx_array_free(contiguous);
        return MLX_NIF_ERROR(env, "failed to get array data");
    }

    ERL_NIF_TERM bin_term;
    unsigned char *buf = enif_make_new_binary(env, nbytes, &bin_term);
    memcpy(buf, data, nbytes);
    mlx_array_free(contiguous);
    return MLX_NIF_OK(env, bin_term);
}

// ============================================================
// NIF: Macros for op patterns
// ============================================================

#define DEF_UNARY_OP(nif_name, mlx_func)                                       \
static ERL_NIF_TERM nif_name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) { \
    (void)argc;                                                                \
    MlxArrayResource *_a; MlxStreamResource *_s;                               \
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&_a) ||   \
        !enif_get_resource(env, argv[1], MLX_STREAM_RESOURCE, (void**)&_s))    \
        return MLX_NIF_ERROR(env, "bad argument");                             \
    mlx_array _r = mlx_array_new();                                            \
    int _ret = mlx_func(&_r, _a->inner, _s->inner);                           \
    if (_ret != 0) return MLX_NIF_ERROR(env, last_error_msg);                  \
    return wrap_array(env, _r);                                                \
}

#define DEF_BINARY_OP(nif_name, mlx_func)                                      \
static ERL_NIF_TERM nif_name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) { \
    (void)argc;                                                                \
    MlxArrayResource *_a, *_b; MlxStreamResource *_s;                          \
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&_a) ||   \
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&_b) ||   \
        !enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&_s))    \
        return MLX_NIF_ERROR(env, "bad argument");                             \
    mlx_array _r = mlx_array_new();                                            \
    int _ret = mlx_func(&_r, _a->inner, _b->inner, _s->inner);                \
    if (_ret != 0) return MLX_NIF_ERROR(env, last_error_msg);                  \
    return wrap_array(env, _r);                                                \
}

// Reduction: reduce(arr, axes_list_or_nil, keepdims_bool, stream)
#define DEF_REDUCE_OP(nif_name, mlx_all, mlx_axes)                             \
static ERL_NIF_TERM nif_name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) { \
    (void)argc;                                                                \
    MlxArrayResource *_a; MlxStreamResource *_s;                               \
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&_a) ||   \
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&_s))    \
        return MLX_NIF_ERROR(env, "bad argument");                             \
    int _kd = (enif_compare(argv[2], ATOM_TRUE) == 0);                         \
    mlx_array _r = mlx_array_new(); int _ret;                                  \
    if (enif_compare(argv[1], ATOM_NIL) == 0) {                                \
        _ret = mlx_all(&_r, _a->inner, _kd, _s->inner);                       \
    } else {                                                                   \
        int _axes[16]; int _nax;                                               \
        if (!get_int_list(env, argv[1], _axes, &_nax))                         \
            return MLX_NIF_ERROR(env, "expected axes list");                    \
        _ret = mlx_axes(&_r, _a->inner, _axes, (size_t)_nax, _kd, _s->inner); \
    }                                                                          \
    if (_ret != 0) return MLX_NIF_ERROR(env, last_error_msg);                  \
    return wrap_array(env, _r);                                                \
}

// ============================================================
// NIF: Unary operations (arr, stream) -> {:ok, arr}
// ============================================================

DEF_UNARY_OP(nif_negative,    mlx_negative)
DEF_UNARY_OP(nif_abs,         mlx_abs)
DEF_UNARY_OP(nif_sign,        mlx_sign)
DEF_UNARY_OP(nif_ceil,        mlx_ceil)
DEF_UNARY_OP(nif_floor,       mlx_floor)
// mlx_round takes a decimals argument, so we can't use DEF_UNARY_OP
static ERL_NIF_TERM nif_round(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *_a; MlxStreamResource *_s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&_a) ||
        !enif_get_resource(env, argv[1], MLX_STREAM_RESOURCE, (void**)&_s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array _r = mlx_array_new();
    int _ret = mlx_round(&_r, _a->inner, 0, _s->inner);
    if (_ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, _r);
}
DEF_UNARY_OP(nif_exp,         mlx_exp)
DEF_UNARY_OP(nif_expm1,       mlx_expm1)
DEF_UNARY_OP(nif_log,         mlx_log)
DEF_UNARY_OP(nif_log2,        mlx_log2)
DEF_UNARY_OP(nif_log10,       mlx_log10)
DEF_UNARY_OP(nif_log1p,       mlx_log1p)
DEF_UNARY_OP(nif_sqrt,        mlx_sqrt)
DEF_UNARY_OP(nif_reciprocal,  mlx_reciprocal)
DEF_UNARY_OP(nif_sin,         mlx_sin)
DEF_UNARY_OP(nif_cos,         mlx_cos)
DEF_UNARY_OP(nif_tan,         mlx_tan)
DEF_UNARY_OP(nif_arcsin,      mlx_arcsin)
DEF_UNARY_OP(nif_arccos,      mlx_arccos)
DEF_UNARY_OP(nif_arctan,      mlx_arctan)
DEF_UNARY_OP(nif_sinh,        mlx_sinh)
DEF_UNARY_OP(nif_cosh,        mlx_cosh)
DEF_UNARY_OP(nif_tanh,        mlx_tanh)
DEF_UNARY_OP(nif_arcsinh,     mlx_arcsinh)
DEF_UNARY_OP(nif_arccosh,     mlx_arccosh)
DEF_UNARY_OP(nif_arctanh,     mlx_arctanh)
DEF_UNARY_OP(nif_erf,         mlx_erf)
DEF_UNARY_OP(nif_erfinv,      mlx_erfinv)
DEF_UNARY_OP(nif_logical_not, mlx_logical_not)
DEF_UNARY_OP(nif_isnan,       mlx_isnan)
DEF_UNARY_OP(nif_isinf,       mlx_isinf)
DEF_UNARY_OP(nif_sigmoid,     mlx_sigmoid)
DEF_UNARY_OP(nif_bitwise_invert, mlx_bitwise_invert)
DEF_UNARY_OP(nif_conjugate,   mlx_conjugate)
DEF_UNARY_OP(nif_real_op,     mlx_real)
DEF_UNARY_OP(nif_imag_op,     mlx_imag)

// Wave 1: New unary ops
DEF_UNARY_OP(nif_rsqrt,        mlx_rsqrt)
DEF_UNARY_OP(nif_square,       mlx_square)
DEF_UNARY_OP(nif_degrees,      mlx_degrees)
DEF_UNARY_OP(nif_radians,      mlx_radians)
DEF_UNARY_OP(nif_isfinite,     mlx_isfinite)
DEF_UNARY_OP(nif_isneginf,     mlx_isneginf)
DEF_UNARY_OP(nif_isposinf,     mlx_isposinf)
DEF_UNARY_OP(nif_copy,         mlx_copy)
DEF_UNARY_OP(nif_stop_gradient, mlx_stop_gradient)
DEF_UNARY_OP(nif_ones_like,    mlx_ones_like)
DEF_UNARY_OP(nif_zeros_like,   mlx_zeros_like)

// ============================================================
// NIF: Binary operations (a, b, stream) -> {:ok, arr}
// ============================================================

DEF_BINARY_OP(nif_add,           mlx_add)
DEF_BINARY_OP(nif_subtract,      mlx_subtract)
DEF_BINARY_OP(nif_multiply,      mlx_multiply)
DEF_BINARY_OP(nif_divide,        mlx_divide)
DEF_BINARY_OP(nif_floor_divide,  mlx_floor_divide)
DEF_BINARY_OP(nif_power,         mlx_power)
DEF_BINARY_OP(nif_logaddexp,     mlx_logaddexp)
DEF_BINARY_OP(nif_arctan2,       mlx_arctan2)
DEF_BINARY_OP(nif_equal,         mlx_equal)
DEF_BINARY_OP(nif_not_equal,     mlx_not_equal)
DEF_BINARY_OP(nif_less,          mlx_less)
DEF_BINARY_OP(nif_less_equal,    mlx_less_equal)
DEF_BINARY_OP(nif_greater,       mlx_greater)
DEF_BINARY_OP(nif_greater_equal, mlx_greater_equal)
DEF_BINARY_OP(nif_logical_and,   mlx_logical_and)
DEF_BINARY_OP(nif_logical_or,    mlx_logical_or)
DEF_BINARY_OP(nif_matmul,        mlx_matmul)
DEF_BINARY_OP(nif_maximum,       mlx_maximum)
DEF_BINARY_OP(nif_minimum,       mlx_minimum)

// Wave 1: New binary ops
DEF_BINARY_OP(nif_remainder,     mlx_remainder)
DEF_BINARY_OP(nif_outer,        mlx_outer)
DEF_BINARY_OP(nif_inner,        mlx_inner)
DEF_BINARY_OP(nif_kron,         mlx_kron)

// ============================================================
// NIF: Reduction operations (arr, axes|nil, keepdims, stream)
// ============================================================

DEF_REDUCE_OP(nif_sum,  mlx_sum,  mlx_sum_axes)
DEF_REDUCE_OP(nif_prod, mlx_prod, mlx_prod_axes)
DEF_REDUCE_OP(nif_mean, mlx_mean, mlx_mean_axes)
DEF_REDUCE_OP(nif_min,  mlx_min,  mlx_min_axes)
DEF_REDUCE_OP(nif_max,  mlx_max,  mlx_max_axes)
DEF_REDUCE_OP(nif_all_op, mlx_all, mlx_all_axes)
DEF_REDUCE_OP(nif_any_op, mlx_any, mlx_any_axes)

// Wave 1: logsumexp reduction
DEF_REDUCE_OP(nif_logsumexp, mlx_logsumexp, mlx_logsumexp_axes)

// Wave 1: Cumulative ops macro
// cumulative(arr, axis, reverse_bool, inclusive_bool, stream) -> {:ok, arr}
#define DEF_CUMULATIVE_OP(nif_name, mlx_func)                                  \
static ERL_NIF_TERM nif_name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) { \
    (void)argc;                                                                \
    MlxArrayResource *_a; MlxStreamResource *_s;                               \
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&_a) ||   \
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&_s))    \
        return MLX_NIF_ERROR(env, "bad argument");                             \
    int _axis;                                                                 \
    if (!enif_get_int(env, argv[1], &_axis))                                   \
        return MLX_NIF_ERROR(env, "expected int axis");                         \
    int _reverse = (enif_compare(argv[2], ATOM_TRUE) == 0);                    \
    int _inclusive = (enif_compare(argv[3], ATOM_TRUE) == 0);                   \
    mlx_array _r = mlx_array_new(); int _ret;                                  \
    _ret = mlx_func(&_r, _a->inner, _axis, _reverse, _inclusive, _s->inner);   \
    if (_ret != 0) return MLX_NIF_ERROR(env, last_error_msg);                  \
    return wrap_array(env, _r);                                                \
}

DEF_CUMULATIVE_OP(nif_cumsum,  mlx_cumsum)
DEF_CUMULATIVE_OP(nif_cumprod, mlx_cumprod)
DEF_CUMULATIVE_OP(nif_cummax,  mlx_cummax)
DEF_CUMULATIVE_OP(nif_cummin,  mlx_cummin)

// Wave 1: Std/Var reductions (extra ddof param)
// std(arr, axes|nil, keepdims, ddof, stream) -> {:ok, arr}
#define DEF_REDUCE_DDOF_OP(nif_name, mlx_all, mlx_axes)                       \
static ERL_NIF_TERM nif_name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) { \
    (void)argc;                                                                \
    MlxArrayResource *_a; MlxStreamResource *_s;                               \
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&_a) ||   \
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&_s))    \
        return MLX_NIF_ERROR(env, "bad argument");                             \
    int _kd = (enif_compare(argv[2], ATOM_TRUE) == 0);                         \
    int _ddof;                                                                 \
    if (!enif_get_int(env, argv[3], &_ddof))                                   \
        return MLX_NIF_ERROR(env, "expected int ddof");                         \
    mlx_array _r = mlx_array_new(); int _ret;                                  \
    if (enif_compare(argv[1], ATOM_NIL) == 0) {                                \
        _ret = mlx_all(&_r, _a->inner, _kd, _ddof, _s->inner);                \
    } else {                                                                   \
        int _axes[16]; int _nax;                                               \
        if (!get_int_list(env, argv[1], _axes, &_nax))                         \
            return MLX_NIF_ERROR(env, "expected axes list");                    \
        _ret = mlx_axes(&_r, _a->inner, _axes, (size_t)_nax, _kd, _ddof, _s->inner); \
    }                                                                          \
    if (_ret != 0) return MLX_NIF_ERROR(env, last_error_msg);                  \
    return wrap_array(env, _r);                                                \
}

DEF_REDUCE_DDOF_OP(nif_std, mlx_std, mlx_std_axes)
DEF_REDUCE_DDOF_OP(nif_var, mlx_var, mlx_var_axes)

// argmax(arr, axis, keepdims, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_argmax(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    int axis;
    if (!enif_get_int(env, argv[1], &axis))
        return MLX_NIF_ERROR(env, "expected int axis");

    int keepdims = (enif_compare(argv[2], ATOM_TRUE) == 0);

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_argmax_axis(&result, a->inner, axis, keepdims, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// argmin(arr, axis, keepdims, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_argmin(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    int axis;
    if (!enif_get_int(env, argv[1], &axis))
        return MLX_NIF_ERROR(env, "expected int axis");

    int keepdims = (enif_compare(argv[2], ATOM_TRUE) == 0);

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_argmin_axis(&result, a->inner, axis, keepdims, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ============================================================
// NIF: Shape operations
// ============================================================

// reshape(arr, shape_list, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_reshape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    int shape[16], ndim;
    if (!get_int_list(env, argv[1], shape, &ndim))
        return MLX_NIF_ERROR(env, "expected shape list");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_reshape(&result, a->inner, shape, ndim, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// transpose(arr, axes_list, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_transpose(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret;

    if (enif_compare(argv[1], ATOM_NIL) == 0) {
        ret = mlx_transpose(&result, a->inner, s->inner);
    } else {
        int axes[16], nax;
        if (!get_int_list(env, argv[1], axes, &nax))
            return MLX_NIF_ERROR(env, "expected axes list");
        ret = mlx_transpose_axes(&result, a->inner, axes, (size_t)nax, s->inner);
    }

    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// squeeze(arr, axes_list_or_nil, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_squeeze(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret;

    if (enif_compare(argv[1], ATOM_NIL) == 0) {
        ret = mlx_squeeze(&result, a->inner, s->inner);
    } else {
        int axes[16], nax;
        if (!get_int_list(env, argv[1], axes, &nax))
            return MLX_NIF_ERROR(env, "expected axes list");
        ret = mlx_squeeze_axes(&result, a->inner, axes, (size_t)nax, s->inner);
    }

    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// expand_dims(arr, axis, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_expand_dims(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    int axis;
    if (!enif_get_int(env, argv[1], &axis))
        return MLX_NIF_ERROR(env, "expected int axis");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    int axes_arr[1] = {axis};
    mlx_array result = mlx_array_new();
    int ret = mlx_expand_dims_axes(&result, a->inner, axes_arr, 1, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// broadcast_to(arr, shape_list, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_broadcast_to(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    int shape[16], ndim;
    if (!get_int_list(env, argv[1], shape, &ndim))
        return MLX_NIF_ERROR(env, "expected shape list");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_broadcast_to(&result, a->inner, shape, ndim, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// flatten(arr, start_axis, end_axis, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_flatten(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    int start_axis, end_axis;
    if (!enif_get_int(env, argv[1], &start_axis) ||
        !enif_get_int(env, argv[2], &end_axis))
        return MLX_NIF_ERROR(env, "expected int axes");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_flatten(&result, a->inner, start_axis, end_axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// slice(arr, start_list, stop_list, strides_list, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_slice(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    int start[16], stop[16], strides[16];
    int n_start, n_stop, n_strides;
    if (!get_int_list(env, argv[1], start, &n_start) ||
        !get_int_list(env, argv[2], stop, &n_stop) ||
        !get_int_list(env, argv[3], strides, &n_strides))
        return MLX_NIF_ERROR(env, "expected int lists for start/stop/strides");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_slice(&result, a->inner, start, n_start, stop, n_stop,
                        strides, n_strides, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ============================================================
// NIF: Type conversion
// ============================================================

// as_type(arr, dtype_atom, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_as_type(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    mlx_dtype dtype;
    if (!dtype_from_atom(env, argv[1], &dtype))
        return MLX_NIF_ERROR(env, "invalid dtype");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_astype(&result, a->inner, dtype, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ============================================================
// NIF: Selection
// ============================================================

// where(condition, x, y, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_where(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *cond_r, *x_r, *y_r;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&cond_r) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&x_r) ||
        !enif_get_resource(env, argv[2], MLX_ARRAY_RESOURCE, (void**)&y_r) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_where(&result, cond_r->inner, x_r->inner, y_r->inner, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ============================================================
// NIF: Additional binary ops (bitwise, shifts)
// ============================================================

DEF_BINARY_OP(nif_bitwise_and,   mlx_bitwise_and)
DEF_BINARY_OP(nif_bitwise_or,    mlx_bitwise_or)
DEF_BINARY_OP(nif_bitwise_xor,   mlx_bitwise_xor)
DEF_BINARY_OP(nif_left_shift,    mlx_left_shift)
DEF_BINARY_OP(nif_right_shift,   mlx_right_shift)

// ============================================================
// NIF: Clip, Sort, Argsort, Take, Triangular, Pad, Repeat, Tile
// ============================================================

// clip(arr, min_arr, max_arr, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_clip(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *a_min, *a_max;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&a_min) ||
        !enif_get_resource(env, argv[2], MLX_ARRAY_RESOURCE, (void**)&a_max) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_clip(&result, a->inner, a_min->inner, a_max->inner, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// sort(arr, axis, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_sort(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    int axis;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &axis) ||
        !enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_sort_axis(&result, a->inner, axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// argsort(arr, axis, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_argsort(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    int axis;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &axis) ||
        !enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_argsort_axis(&result, a->inner, axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// take(arr, indices, axis, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_take(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *indices;
    int axis;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&indices) ||
        !enif_get_int(env, argv[2], &axis) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_take_axis(&result, a->inner, indices->inner, axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// take_along_axis(arr, indices, axis, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_take_along_axis(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *indices;
    int axis;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&indices) ||
        !enif_get_int(env, argv[2], &axis) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_take_along_axis(&result, a->inner, indices->inner, axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// triu(arr, k, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_triu(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    int k;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &k) ||
        !enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_triu(&result, a->inner, k, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// tril(arr, k, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_tril(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    int k;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &k) ||
        !enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_tril(&result, a->inner, k, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// pad(arr, axes_list, low_pad_list, high_pad_list, pad_value_arr, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_pad(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *pad_val;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[4], MLX_ARRAY_RESOURCE, (void**)&pad_val) ||
        !enif_get_resource(env, argv[5], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    int axes[16], low_pad[16], high_pad[16];
    int n_axes, n_low, n_high;
    if (!get_int_list(env, argv[1], axes, &n_axes) ||
        !get_int_list(env, argv[2], low_pad, &n_low) ||
        !get_int_list(env, argv[3], high_pad, &n_high))
        return MLX_NIF_ERROR(env, "expected int lists for axes/low_pad/high_pad");

    mlx_array result = mlx_array_new();
    int ret = mlx_pad(&result, a->inner, axes, (size_t)n_axes,
                      low_pad, (size_t)n_low, high_pad, (size_t)n_high,
                      pad_val->inner, "constant", s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// repeat(arr, repeats, axis, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_repeat(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    int repeats, axis;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &repeats) ||
        !enif_get_int(env, argv[2], &axis) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_repeat_axis(&result, a->inner, repeats, axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// tile(arr, reps_list, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_tile(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    int reps[16], n_reps;
    if (!get_int_list(env, argv[1], reps, &n_reps))
        return MLX_NIF_ERROR(env, "expected int list for reps");

    mlx_array result = mlx_array_new();
    int ret = mlx_tile(&result, a->inner, reps, (size_t)n_reps, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// concatenate(arr_ref_list, axis, stream) -> {:ok, arr}
// Takes a list of array refs, builds a vector_array, calls mlx_concatenate
static ERL_NIF_TERM nif_concatenate(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    int axis;
    if (!enif_get_int(env, argv[1], &axis))
        return MLX_NIF_ERROR(env, "expected int axis");

    // Build vector_array from list of array refs
    unsigned int list_len;
    if (!enif_get_list_length(env, argv[0], &list_len))
        return MLX_NIF_ERROR(env, "expected list of arrays");

    mlx_vector_array vec = mlx_vector_array_new();
    if (!vec.ctx) return MLX_NIF_ERROR(env, "failed to create vector_array");

    ERL_NIF_TERM head, tail = argv[0];
    for (unsigned int i = 0; i < list_len; i++) {
        if (!enif_get_list_cell(env, tail, &head, &tail)) {
            mlx_vector_array_free(vec);
            return MLX_NIF_ERROR(env, "bad list element");
        }
        MlxArrayResource *arr;
        if (!enif_get_resource(env, head, MLX_ARRAY_RESOURCE, (void**)&arr)) {
            mlx_vector_array_free(vec);
            return MLX_NIF_ERROR(env, "expected array in list");
        }
        int ret = mlx_vector_array_append_value(vec, arr->inner);
        if (ret != 0) {
            mlx_vector_array_free(vec);
            return MLX_NIF_ERROR(env, last_error_msg);
        }
    }

    mlx_array result = mlx_array_new();
    int ret = mlx_concatenate_axis(&result, vec, axis, s->inner);
    mlx_vector_array_free(vec);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// stack(arr_ref_list, axis, stream) -> {:ok, arr}
// Takes a list of array refs, builds a vector_array, calls mlx_stack
static ERL_NIF_TERM nif_stack(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    int axis;
    if (!enif_get_int(env, argv[1], &axis))
        return MLX_NIF_ERROR(env, "expected int axis");

    unsigned int list_len;
    if (!enif_get_list_length(env, argv[0], &list_len))
        return MLX_NIF_ERROR(env, "expected list of arrays");

    mlx_vector_array vec = mlx_vector_array_new();
    if (!vec.ctx) return MLX_NIF_ERROR(env, "failed to create vector_array");

    ERL_NIF_TERM head, tail = argv[0];
    for (unsigned int i = 0; i < list_len; i++) {
        if (!enif_get_list_cell(env, tail, &head, &tail)) {
            mlx_vector_array_free(vec);
            return MLX_NIF_ERROR(env, "bad list element");
        }
        MlxArrayResource *arr;
        if (!enif_get_resource(env, head, MLX_ARRAY_RESOURCE, (void**)&arr)) {
            mlx_vector_array_free(vec);
            return MLX_NIF_ERROR(env, "expected array in list");
        }
        int ret = mlx_vector_array_append_value(vec, arr->inner);
        if (ret != 0) {
            mlx_vector_array_free(vec);
            return MLX_NIF_ERROR(env, last_error_msg);
        }
    }

    mlx_array result = mlx_array_new();
    int ret = mlx_stack_axis(&result, vec, axis, s->inner);
    mlx_vector_array_free(vec);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// slice_update(src, update, start_list, stop_list, strides_list, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_slice_update(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *src, *update;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&src))
        return MLX_NIF_ERROR(env, "expected source array");
    if (!enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&update))
        return MLX_NIF_ERROR(env, "expected update array");
    if (!enif_get_resource(env, argv[5], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    int start[16], stop[16], strides[16];
    int n_start, n_stop, n_strides;
    if (!get_int_list(env, argv[2], start, &n_start))
        return MLX_NIF_ERROR(env, "expected start list");
    if (!get_int_list(env, argv[3], stop, &n_stop))
        return MLX_NIF_ERROR(env, "expected stop list");
    if (!get_int_list(env, argv[4], strides, &n_strides))
        return MLX_NIF_ERROR(env, "expected strides list");

    mlx_array result = mlx_array_new();
    int ret = mlx_slice_update(&result, src->inner, update->inner,
                               start, (size_t)n_start,
                               stop, (size_t)n_stop,
                               strides, (size_t)n_strides,
                               s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// view(arr, dtype_atom, stream) -> {:ok, arr}
// Bitcast: reinterpret array bits as a different dtype
static ERL_NIF_TERM nif_view(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "expected array");

    mlx_dtype dtype;
    if (!dtype_from_atom(env, argv[1], &dtype))
        return MLX_NIF_ERROR(env, "invalid dtype");

    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_view(&result, a->inner, dtype, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ============================================================
// NIF: FFT operations
// ============================================================

// fft(arr, n, axis, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_fft(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    int n, axis;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &n) ||
        !enif_get_int(env, argv[2], &axis) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_fft_fft(&result, a->inner, n, axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ifft(arr, n, axis, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_ifft(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    int n, axis;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &n) ||
        !enif_get_int(env, argv[2], &axis) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_fft_ifft(&result, a->inner, n, axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ============================================================
// NIF: Gather / Scatter operations
// ============================================================

// gather(arr, indices_list, axes_list, slice_sizes_list, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_gather(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    // Build vector_array of indices from list of array refs
    unsigned int list_len;
    if (!enif_get_list_length(env, argv[1], &list_len))
        return MLX_NIF_ERROR(env, "expected list of index arrays");

    mlx_vector_array indices_vec = mlx_vector_array_new();
    if (!indices_vec.ctx) return MLX_NIF_ERROR(env, "failed to create vector_array");

    ERL_NIF_TERM head, tail = argv[1];
    for (unsigned int i = 0; i < list_len; i++) {
        if (!enif_get_list_cell(env, tail, &head, &tail)) {
            mlx_vector_array_free(indices_vec);
            return MLX_NIF_ERROR(env, "bad list element");
        }
        MlxArrayResource *idx;
        if (!enif_get_resource(env, head, MLX_ARRAY_RESOURCE, (void**)&idx)) {
            mlx_vector_array_free(indices_vec);
            return MLX_NIF_ERROR(env, "expected array in indices list");
        }
        int ret = mlx_vector_array_append_value(indices_vec, idx->inner);
        if (ret != 0) {
            mlx_vector_array_free(indices_vec);
            return MLX_NIF_ERROR(env, last_error_msg);
        }
    }

    int axes[16], slice_sizes[16];
    int n_axes, n_slice_sizes;
    if (!get_int_list(env, argv[2], axes, &n_axes) ||
        !get_int_list(env, argv[3], slice_sizes, &n_slice_sizes)) {
        mlx_vector_array_free(indices_vec);
        return MLX_NIF_ERROR(env, "expected int lists for axes/slice_sizes");
    }

    mlx_array result = mlx_array_new();
    int ret = mlx_gather(&result, a->inner, indices_vec,
                         axes, (size_t)n_axes,
                         slice_sizes, (size_t)n_slice_sizes, s->inner);
    mlx_vector_array_free(indices_vec);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// scatter(arr, indices_list, updates, axes_list, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_scatter(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *updates;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[2], MLX_ARRAY_RESOURCE, (void**)&updates) ||
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    // Build vector_array of indices from list of array refs
    unsigned int list_len;
    if (!enif_get_list_length(env, argv[1], &list_len))
        return MLX_NIF_ERROR(env, "expected list of index arrays");

    mlx_vector_array indices_vec = mlx_vector_array_new();
    if (!indices_vec.ctx) return MLX_NIF_ERROR(env, "failed to create vector_array");

    ERL_NIF_TERM head, tail = argv[1];
    for (unsigned int i = 0; i < list_len; i++) {
        if (!enif_get_list_cell(env, tail, &head, &tail)) {
            mlx_vector_array_free(indices_vec);
            return MLX_NIF_ERROR(env, "bad list element");
        }
        MlxArrayResource *idx;
        if (!enif_get_resource(env, head, MLX_ARRAY_RESOURCE, (void**)&idx)) {
            mlx_vector_array_free(indices_vec);
            return MLX_NIF_ERROR(env, "expected array in indices list");
        }
        int ret = mlx_vector_array_append_value(indices_vec, idx->inner);
        if (ret != 0) {
            mlx_vector_array_free(indices_vec);
            return MLX_NIF_ERROR(env, last_error_msg);
        }
    }

    int axes[16];
    int n_axes;
    if (!get_int_list(env, argv[3], axes, &n_axes)) {
        mlx_vector_array_free(indices_vec);
        return MLX_NIF_ERROR(env, "expected int list for axes");
    }

    mlx_array result = mlx_array_new();
    int ret = mlx_scatter(&result, a->inner, indices_vec, updates->inner,
                          axes, (size_t)n_axes, s->inner);
    mlx_vector_array_free(indices_vec);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// scatter_add(arr, indices_list, updates, axes_list, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_scatter_add(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *updates;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[2], MLX_ARRAY_RESOURCE, (void**)&updates) ||
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    unsigned int list_len;
    if (!enif_get_list_length(env, argv[1], &list_len))
        return MLX_NIF_ERROR(env, "expected list of index arrays");

    mlx_vector_array indices_vec = mlx_vector_array_new();
    if (!indices_vec.ctx) return MLX_NIF_ERROR(env, "failed to create vector_array");

    ERL_NIF_TERM head, tail = argv[1];
    for (unsigned int i = 0; i < list_len; i++) {
        if (!enif_get_list_cell(env, tail, &head, &tail)) {
            mlx_vector_array_free(indices_vec);
            return MLX_NIF_ERROR(env, "bad list element");
        }
        MlxArrayResource *idx;
        if (!enif_get_resource(env, head, MLX_ARRAY_RESOURCE, (void**)&idx)) {
            mlx_vector_array_free(indices_vec);
            return MLX_NIF_ERROR(env, "expected array in indices list");
        }
        int ret = mlx_vector_array_append_value(indices_vec, idx->inner);
        if (ret != 0) {
            mlx_vector_array_free(indices_vec);
            return MLX_NIF_ERROR(env, last_error_msg);
        }
    }

    int axes[16];
    int n_axes;
    if (!get_int_list(env, argv[3], axes, &n_axes)) {
        mlx_vector_array_free(indices_vec);
        return MLX_NIF_ERROR(env, "expected int list for axes");
    }

    mlx_array result = mlx_array_new();
    int ret = mlx_scatter_add(&result, a->inner, indices_vec, updates->inner,
                              axes, (size_t)n_axes, s->inner);
    mlx_vector_array_free(indices_vec);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ============================================================
// NIF: Wave 1 - Shape/Selection/Comparison/Matrix ops
// ============================================================

// moveaxis(arr, src, dst, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_moveaxis(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    int src, dst;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &src) ||
        !enif_get_int(env, argv[2], &dst) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_moveaxis(&result, a->inner, src, dst, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// swapaxes(arr, ax1, ax2, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_swapaxes(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    int ax1, ax2;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &ax1) ||
        !enif_get_int(env, argv[2], &ax2) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_swapaxes(&result, a->inner, ax1, ax2, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// diagonal(arr, offset, ax1, ax2, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_diagonal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    int offset, ax1, ax2;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &offset) ||
        !enif_get_int(env, argv[2], &ax1) ||
        !enif_get_int(env, argv[3], &ax2) ||
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_diagonal(&result, a->inner, offset, ax1, ax2, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// diag(arr, k, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_diag(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    int k;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &k) ||
        !enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_diag(&result, a->inner, k, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// roll(arr, shift, axes_list_or_nil, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_roll(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    int shift;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &shift) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    int shifts[1] = {shift};
    mlx_array result = mlx_array_new();
    int ret;
    if (enif_compare(argv[2], ATOM_NIL) == 0) {
        ret = mlx_roll(&result, a->inner, shifts, 1, s->inner);
    } else {
        int axes[16]; int nax;
        if (!get_int_list(env, argv[2], axes, &nax))
            return MLX_NIF_ERROR(env, "expected axes list");
        ret = mlx_roll_axes(&result, a->inner, shifts, 1, axes, (size_t)nax, s->inner);
    }
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// unflatten(arr, axis, shape_list, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_unflatten(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    int axis;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &axis) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    int shape[16]; int ndim;
    if (!get_int_list(env, argv[2], shape, &ndim))
        return MLX_NIF_ERROR(env, "expected shape list");
    mlx_array result = mlx_array_new();
    int ret = mlx_unflatten(&result, a->inner, axis, shape, (size_t)ndim, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// Helper: get int64 list
static int get_int64_list(ErlNifEnv* env, ERL_NIF_TERM list, int64_t* out, int* len) {
    unsigned int ulen;
    if (!enif_get_list_length(env, list, &ulen)) return 0;
    *len = (int)ulen;
    ERL_NIF_TERM head, tail = list;
    for (unsigned int i = 0; i < ulen; i++) {
        if (!enif_get_list_cell(env, tail, &head, &tail)) return 0;
        ErlNifSInt64 val;
        if (!enif_get_int64(env, head, &val)) return 0;
        out[i] = (int64_t)val;
    }
    return 1;
}

// as_strided(arr, shape_list, strides_list, offset, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_as_strided(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    int shape[16]; int ndim;
    if (!get_int_list(env, argv[1], shape, &ndim))
        return MLX_NIF_ERROR(env, "expected shape list");
    int64_t strides[16]; int nstrides;
    if (!get_int64_list(env, argv[2], strides, &nstrides))
        return MLX_NIF_ERROR(env, "expected strides list");
    ErlNifUInt64 offset;
    if (!enif_get_uint64(env, argv[3], &offset))
        return MLX_NIF_ERROR(env, "expected uint offset");
    mlx_array result = mlx_array_new();
    int ret = mlx_as_strided(&result, a->inner, shape, (size_t)ndim,
                             strides, (size_t)nstrides, (size_t)offset, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// identity(n, dtype, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_identity(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    int n;
    mlx_dtype dtype;
    MlxStreamResource *s;
    if (!enif_get_int(env, argv[0], &n) ||
        !dtype_from_atom(env, argv[1], &dtype) ||
        !enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_identity(&result, n, dtype, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// tri(n, m, k, dtype, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_tri(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    int n, m, k;
    mlx_dtype dtype;
    MlxStreamResource *s;
    if (!enif_get_int(env, argv[0], &n) ||
        !enif_get_int(env, argv[1], &m) ||
        !enif_get_int(env, argv[2], &k) ||
        !dtype_from_atom(env, argv[3], &dtype) ||
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_tri(&result, n, m, k, dtype, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// topk(arr, k, axis, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_topk(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    int k, axis;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &k) ||
        !enif_get_int(env, argv[2], &axis) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_topk_axis(&result, a->inner, k, axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// partition(arr, kth, axis, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_partition(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    int kth, axis;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &kth) ||
        !enif_get_int(env, argv[2], &axis) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_partition_axis(&result, a->inner, kth, axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// argpartition(arr, kth, axis, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_argpartition(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    int kth, axis;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &kth) ||
        !enif_get_int(env, argv[2], &axis) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_argpartition_axis(&result, a->inner, kth, axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// put_along_axis(arr, indices, values, axis, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_put_along_axis(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *indices, *values; MlxStreamResource *s;
    int axis;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&indices) ||
        !enif_get_resource(env, argv[2], MLX_ARRAY_RESOURCE, (void**)&values) ||
        !enif_get_int(env, argv[3], &axis) ||
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_put_along_axis(&result, a->inner, indices->inner, values->inner, axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// allclose(a, b, rtol, atol, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_allclose(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *b; MlxStreamResource *s;
    double rtol, atol;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&b) ||
        !enif_get_double(env, argv[2], &rtol) ||
        !enif_get_double(env, argv[3], &atol) ||
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_allclose(&result, a->inner, b->inner, rtol, atol, false, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// isclose(a, b, rtol, atol, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_isclose(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *b; MlxStreamResource *s;
    double rtol, atol;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&b) ||
        !enif_get_double(env, argv[2], &rtol) ||
        !enif_get_double(env, argv[3], &atol) ||
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_isclose(&result, a->inner, b->inner, rtol, atol, false, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// array_equal(a, b, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_array_equal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *b; MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&b) ||
        !enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_array_equal(&result, a->inner, b->inner, false, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// nan_to_num(arr, nan, posinf_or_nil, neginf_or_nil, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_nan_to_num(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    double nan_val;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_double(env, argv[1], &nan_val) ||
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_optional_float posinf = {0, false};
    mlx_optional_float neginf = {0, false};
    double tmp;
    if (enif_compare(argv[2], ATOM_NIL) != 0) {
        if (!enif_get_double(env, argv[2], &tmp))
            return MLX_NIF_ERROR(env, "expected double or nil for posinf");
        posinf.value = (float)tmp;
        posinf.has_value = true;
    }
    if (enif_compare(argv[3], ATOM_NIL) != 0) {
        if (!enif_get_double(env, argv[3], &tmp))
            return MLX_NIF_ERROR(env, "expected double or nil for neginf");
        neginf.value = (float)tmp;
        neginf.has_value = true;
    }

    mlx_array result = mlx_array_new();
    int ret = mlx_nan_to_num(&result, a->inner, (float)nan_val, posinf, neginf, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// number_of_elements(arr, axes_list, inverted_bool, dtype_atom, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_number_of_elements(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    int axes[16]; int nax;
    if (!get_int_list(env, argv[1], axes, &nax))
        return MLX_NIF_ERROR(env, "expected axes list");
    int inverted = (enif_compare(argv[2], ATOM_TRUE) == 0);
    mlx_dtype dtype;
    if (!dtype_from_atom(env, argv[3], &dtype))
        return MLX_NIF_ERROR(env, "invalid dtype");
    mlx_array result = mlx_array_new();
    int ret = mlx_number_of_elements(&result, a->inner, axes, (size_t)nax, inverted, dtype, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// softmax(arr, axes_list, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_softmax(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret;
    if (enif_compare(argv[1], ATOM_NIL) == 0) {
        ret = mlx_softmax(&result, a->inner, false, s->inner);
    } else {
        int axes[16]; int nax;
        if (!get_int_list(env, argv[1], axes, &nax))
            return MLX_NIF_ERROR(env, "expected axes list");
        ret = mlx_softmax_axes(&result, a->inner, axes, (size_t)nax, false, s->inner);
    }
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// tensordot(a, b, axes_a_list, axes_b_list, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_tensordot(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *b; MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&b) ||
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    int axes_a[16], axes_b[16]; int na, nb;
    if (!get_int_list(env, argv[2], axes_a, &na) ||
        !get_int_list(env, argv[3], axes_b, &nb))
        return MLX_NIF_ERROR(env, "expected axes lists");
    mlx_array result = mlx_array_new();
    int ret = mlx_tensordot(&result, a->inner, b->inner,
                            axes_a, (size_t)na, axes_b, (size_t)nb, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// addmm(c, a, b, alpha, beta, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_addmm(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *c_arr, *a, *b; MlxStreamResource *s;
    double alpha, beta;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&c_arr) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[2], MLX_ARRAY_RESOURCE, (void**)&b) ||
        !enif_get_double(env, argv[3], &alpha) ||
        !enif_get_double(env, argv[4], &beta) ||
        !enif_get_resource(env, argv[5], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_addmm(&result, c_arr->inner, a->inner, b->inner,
                        (float)alpha, (float)beta, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// trace(arr, offset, ax1, ax2, dtype, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_trace(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    int offset, ax1, ax2;
    mlx_dtype dtype;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &offset) ||
        !enif_get_int(env, argv[2], &ax1) ||
        !enif_get_int(env, argv[3], &ax2) ||
        !dtype_from_atom(env, argv[4], &dtype) ||
        !enif_get_resource(env, argv[5], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_trace(&result, a->inner, offset, ax1, ax2, dtype, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// einsum(subscripts_string, operands_list, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_einsum(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    char subscripts[256];
    if (enif_get_string(env, argv[0], subscripts, sizeof(subscripts), ERL_NIF_LATIN1) <= 0)
        return MLX_NIF_ERROR(env, "expected string for subscripts");

    unsigned int list_len;
    if (!enif_get_list_length(env, argv[1], &list_len))
        return MLX_NIF_ERROR(env, "expected list of arrays");

    mlx_vector_array operands = mlx_vector_array_new();
    ERL_NIF_TERM head, tail = argv[1];
    for (unsigned int i = 0; i < list_len; i++) {
        if (!enif_get_list_cell(env, tail, &head, &tail)) {
            mlx_vector_array_free(operands);
            return MLX_NIF_ERROR(env, "bad list element");
        }
        MlxArrayResource *arr;
        if (!enif_get_resource(env, head, MLX_ARRAY_RESOURCE, (void**)&arr)) {
            mlx_vector_array_free(operands);
            return MLX_NIF_ERROR(env, "expected array in operands list");
        }
        mlx_vector_array_append_value(operands, arr->inner);
    }

    mlx_array result = mlx_array_new();
    int ret = mlx_einsum(&result, subscripts, operands, s->inner);
    mlx_vector_array_free(operands);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// block_masked_mm(a, b, block_size, mask_out_or_nil, mask_lhs_or_nil, mask_rhs_or_nil, stream)
static ERL_NIF_TERM nif_block_masked_mm(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *b; MlxStreamResource *s;
    int block_size;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&b) ||
        !enif_get_int(env, argv[2], &block_size) ||
        !enif_get_resource(env, argv[6], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    // Optional mask arrays (nil = no mask)
    mlx_array mask_out = {NULL};
    mlx_array mask_lhs = {NULL};
    mlx_array mask_rhs = {NULL};
    MlxArrayResource *mo, *ml, *mr;
    if (enif_compare(argv[3], ATOM_NIL) != 0) {
        if (!enif_get_resource(env, argv[3], MLX_ARRAY_RESOURCE, (void**)&mo))
            return MLX_NIF_ERROR(env, "expected array or nil for mask_out");
        mask_out = mo->inner;
    }
    if (enif_compare(argv[4], ATOM_NIL) != 0) {
        if (!enif_get_resource(env, argv[4], MLX_ARRAY_RESOURCE, (void**)&ml))
            return MLX_NIF_ERROR(env, "expected array or nil for mask_lhs");
        mask_lhs = ml->inner;
    }
    if (enif_compare(argv[5], ATOM_NIL) != 0) {
        if (!enif_get_resource(env, argv[5], MLX_ARRAY_RESOURCE, (void**)&mr))
            return MLX_NIF_ERROR(env, "expected array or nil for mask_rhs");
        mask_rhs = mr->inner;
    }

    mlx_array result = mlx_array_new();
    int ret = mlx_block_masked_mm(&result, a->inner, b->inner, block_size,
                                   mask_out, mask_lhs, mask_rhs, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// Wave 1: Scatter variants (same pattern as scatter_add)
#define DEF_SCATTER_OP(nif_name, mlx_func)                                     \
static ERL_NIF_TERM nif_name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) { \
    (void)argc;                                                                \
    MlxArrayResource *_a, *_updates; MlxStreamResource *_s;                    \
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&_a) ||   \
        !enif_get_resource(env, argv[2], MLX_ARRAY_RESOURCE, (void**)&_updates) || \
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&_s))    \
        return MLX_NIF_ERROR(env, "bad argument");                             \
    unsigned int _ll;                                                          \
    if (!enif_get_list_length(env, argv[1], &_ll))                             \
        return MLX_NIF_ERROR(env, "expected list of index arrays");            \
    mlx_vector_array _iv = mlx_vector_array_new();                             \
    ERL_NIF_TERM _h, _t = argv[1];                                            \
    for (unsigned int _i = 0; _i < _ll; _i++) {                               \
        if (!enif_get_list_cell(env, _t, &_h, &_t)) {                         \
            mlx_vector_array_free(_iv);                                        \
            return MLX_NIF_ERROR(env, "bad list element");                     \
        }                                                                      \
        MlxArrayResource *_idx;                                                \
        if (!enif_get_resource(env, _h, MLX_ARRAY_RESOURCE, (void**)&_idx)) {  \
            mlx_vector_array_free(_iv);                                        \
            return MLX_NIF_ERROR(env, "expected array in indices list");        \
        }                                                                      \
        mlx_vector_array_append_value(_iv, _idx->inner);                       \
    }                                                                          \
    int _axes[16]; int _nax;                                                   \
    if (!get_int_list(env, argv[3], _axes, &_nax)) {                           \
        mlx_vector_array_free(_iv);                                            \
        return MLX_NIF_ERROR(env, "expected int list for axes");               \
    }                                                                          \
    mlx_array _r = mlx_array_new();                                            \
    int _ret = mlx_func(&_r, _a->inner, _iv, _updates->inner,                 \
                        _axes, (size_t)_nax, _s->inner);                       \
    mlx_vector_array_free(_iv);                                                \
    if (_ret != 0) return MLX_NIF_ERROR(env, last_error_msg);                  \
    return wrap_array(env, _r);                                                \
}

DEF_SCATTER_OP(nif_scatter_max,  mlx_scatter_max)
DEF_SCATTER_OP(nif_scatter_min,  mlx_scatter_min)
DEF_SCATTER_OP(nif_scatter_prod, mlx_scatter_prod)

// ============================================================
// NIF: Convolution
// ============================================================

// conv_general(input, weight, strides, padding_lo, padding_hi,
//              kernel_dilation, input_dilation, groups, flip, stream)
static ERL_NIF_TERM nif_conv_general(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *input, *weight;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&input) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&weight) ||
        !enif_get_resource(env, argv[9], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    int strides[16], padding_lo[16], padding_hi[16];
    int kernel_dilation[16], input_dilation_arr[16];
    int n_strides, n_pad_lo, n_pad_hi, n_kdil, n_idil;

    if (!get_int_list(env, argv[2], strides, &n_strides) ||
        !get_int_list(env, argv[3], padding_lo, &n_pad_lo) ||
        !get_int_list(env, argv[4], padding_hi, &n_pad_hi) ||
        !get_int_list(env, argv[5], kernel_dilation, &n_kdil) ||
        !get_int_list(env, argv[6], input_dilation_arr, &n_idil))
        return MLX_NIF_ERROR(env, "expected int lists for conv params");

    int groups;
    if (!enif_get_int(env, argv[7], &groups))
        return MLX_NIF_ERROR(env, "expected int for groups");

    int flip = (enif_compare(argv[8], ATOM_TRUE) == 0);

    mlx_array result = mlx_array_new();
    int ret = mlx_conv_general(&result, input->inner, weight->inner,
                               strides, (size_t)n_strides,
                               padding_lo, (size_t)n_pad_lo,
                               padding_hi, (size_t)n_pad_hi,
                               kernel_dilation, (size_t)n_kdil,
                               input_dilation_arr, (size_t)n_idil,
                               groups, flip, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ============================================================
// NIF: Split (for to_batched)
// ============================================================

// split_equal_parts(arr, num_splits, axis, stream) -> {:ok, [arr_ref]}
// Returns a list of array refs (one per split part)
static ERL_NIF_TERM nif_split_equal_parts(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    int num_splits, axis;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &num_splits) ||
        !enif_get_int(env, argv[2], &axis) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_vector_array parts = mlx_vector_array_new();
    int ret = mlx_split(&parts, a->inner, num_splits, axis, s->inner);
    if (ret != 0) {
        mlx_vector_array_free(parts);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    // Convert vector_array to list of array refs
    size_t n = mlx_vector_array_size(parts);
    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int i = (int)n - 1; i >= 0; i--) {
        mlx_array part = mlx_array_new();
        int get_ret = mlx_vector_array_get(&part, parts, i);
        if (get_ret != 0) {
            mlx_array_free(part);
            mlx_vector_array_free(parts);
            return MLX_NIF_ERROR(env, last_error_msg);
        }
        MlxArrayResource *res = enif_alloc_resource(MLX_ARRAY_RESOURCE, sizeof(MlxArrayResource));
        res->inner = part;
        ERL_NIF_TERM term = enif_make_resource(env, res);
        enif_release_resource(res);
        list = enif_make_list_cell(env, term, list);
    }

    mlx_vector_array_free(parts);
    return MLX_NIF_OK(env, list);
}

// ============================================================
// NIF: Device and Stream management
// ============================================================

// default_cpu_stream() -> {:ok, stream_ref}
static ERL_NIF_TERM nif_default_cpu_stream(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc; (void)argv;
    mlx_stream s = mlx_default_cpu_stream_new();
    if (!s.ctx) return MLX_NIF_ERROR(env, "failed to get default CPU stream");
    return wrap_stream(env, s);
}

// default_gpu_stream() -> {:ok, stream_ref}
static ERL_NIF_TERM nif_default_gpu_stream(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc; (void)argv;
    mlx_stream s = mlx_default_gpu_stream_new();
    if (!s.ctx) return MLX_NIF_ERROR(env, "failed to get default GPU stream");
    return wrap_stream(env, s);
}

// default_device() -> {:ok, device_ref}
static ERL_NIF_TERM nif_default_device(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc; (void)argv;
    mlx_device d = mlx_device_new();
    int ret = mlx_get_default_device(&d);
    if (ret != 0) return MLX_NIF_ERROR(env, "failed to get default device");
    return wrap_device(env, d);
}

// set_default_device(device_ref) -> :ok
static ERL_NIF_TERM nif_set_default_device(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxDeviceResource *d;
    if (!enif_get_resource(env, argv[0], MLX_DEVICE_RESOURCE, (void**)&d))
        return MLX_NIF_ERROR(env, "expected device");

    int ret = mlx_set_default_device(d->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return ATOM_OK;
}

// device_new(type_atom) -> {:ok, device_ref}
static ERL_NIF_TERM nif_device_new(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    mlx_device_type dtype;
    if (enif_compare(argv[0], ATOM_CPU) == 0) {
        dtype = MLX_CPU;
    } else if (enif_compare(argv[0], ATOM_GPU) == 0) {
        dtype = MLX_GPU;
    } else {
        return MLX_NIF_ERROR(env, "expected :cpu or :gpu");
    }

    mlx_device d = mlx_device_new_type(dtype, 0);
    if (!d.ctx) return MLX_NIF_ERROR(env, "failed to create device");
    return wrap_device(env, d);
}

// synchronize(stream) -> :ok
static ERL_NIF_TERM nif_synchronize(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    int ret = mlx_synchronize(s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return ATOM_OK;
}

// ============================================================
// Wave 2: Random ops
// ============================================================

// Helper: extract optional key (nil atom means null/default key)
static int get_optional_key(ErlNifEnv* env, ERL_NIF_TERM term, mlx_array* out) {
    if (enif_compare(term, ATOM_NIL) == 0) {
        out->ctx = NULL;
        return 1;
    }
    MlxArrayResource *res;
    if (enif_get_resource(env, term, MLX_ARRAY_RESOURCE, (void**)&res)) {
        *out = res->inner;
        return 1;
    }
    return 0;
}

// random_key(seed) -> {:ok, arr}
static ERL_NIF_TERM nif_random_key(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    ErlNifUInt64 seed;
    if (!enif_get_uint64(env, argv[0], &seed))
        return MLX_NIF_ERROR(env, "expected uint64 seed");
    mlx_array result = mlx_array_new();
    int ret = mlx_random_key(&result, (uint64_t)seed);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// random_seed(seed) -> :ok
static ERL_NIF_TERM nif_random_seed(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    ErlNifUInt64 seed;
    if (!enif_get_uint64(env, argv[0], &seed))
        return MLX_NIF_ERROR(env, "expected uint64 seed");
    int ret = mlx_random_seed((uint64_t)seed);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return ATOM_OK;
}

// random_split(key, stream) -> {:ok, {arr, arr}}
static ERL_NIF_TERM nif_random_split(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *key_res;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&key_res) ||
        !enif_get_resource(env, argv[1], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array res0 = mlx_array_new();
    mlx_array res1 = mlx_array_new();
    int ret = mlx_random_split(&res0, &res1, key_res->inner, s->inner);
    if (ret != 0) {
        mlx_array_free(res0);
        mlx_array_free(res1);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    MlxArrayResource *r0 = enif_alloc_resource(MLX_ARRAY_RESOURCE, sizeof(MlxArrayResource));
    r0->inner = res0;
    ERL_NIF_TERM t0 = enif_make_resource(env, r0);
    enif_release_resource(r0);

    MlxArrayResource *r1 = enif_alloc_resource(MLX_ARRAY_RESOURCE, sizeof(MlxArrayResource));
    r1->inner = res1;
    ERL_NIF_TERM t1 = enif_make_resource(env, r1);
    enif_release_resource(r1);

    return MLX_NIF_OK(env, enif_make_tuple2(env, t0, t1));
}

// random_split_num(key, num, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_random_split_num(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *key_res;
    MlxStreamResource *s;
    int num;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&key_res) ||
        !enif_get_int(env, argv[1], &num) ||
        !enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_random_split_num(&result, key_res->inner, num, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// random_uniform(low, high, shape, dtype, key_or_nil, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_random_uniform(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *low, *high;
    MlxStreamResource *s;
    int shape[16], ndim;
    mlx_dtype dtype;
    mlx_array key;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&low) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&high) ||
        !get_int_list(env, argv[2], shape, &ndim) ||
        !dtype_from_atom(env, argv[3], &dtype) ||
        !get_optional_key(env, argv[4], &key) ||
        !enif_get_resource(env, argv[5], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_random_uniform(&result, low->inner, high->inner, shape, ndim, dtype, key, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// random_normal(shape, dtype, loc, scale, key_or_nil, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_random_normal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxStreamResource *s;
    int shape[16], ndim;
    mlx_dtype dtype;
    double loc, scale;
    mlx_array key;
    if (!get_int_list(env, argv[0], shape, &ndim) ||
        !dtype_from_atom(env, argv[1], &dtype) ||
        !enif_get_double(env, argv[2], &loc) ||
        !enif_get_double(env, argv[3], &scale) ||
        !get_optional_key(env, argv[4], &key) ||
        !enif_get_resource(env, argv[5], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_random_normal(&result, shape, ndim, dtype, (float)loc, (float)scale, key, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// random_bernoulli(p, shape, key_or_nil, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_random_bernoulli(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *p;
    MlxStreamResource *s;
    int shape[16], ndim;
    mlx_array key;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&p) ||
        !get_int_list(env, argv[1], shape, &ndim) ||
        !get_optional_key(env, argv[2], &key) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_random_bernoulli(&result, p->inner, shape, ndim, key, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// random_randint(low, high, shape, dtype, key_or_nil, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_random_randint(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *low, *high;
    MlxStreamResource *s;
    int shape[16], ndim;
    mlx_dtype dtype;
    mlx_array key;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&low) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&high) ||
        !get_int_list(env, argv[2], shape, &ndim) ||
        !dtype_from_atom(env, argv[3], &dtype) ||
        !get_optional_key(env, argv[4], &key) ||
        !enif_get_resource(env, argv[5], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_random_randint(&result, low->inner, high->inner, shape, ndim, dtype, key, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// random_truncated_normal(lower, upper, shape, dtype, key_or_nil, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_random_truncated_normal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *lower, *upper;
    MlxStreamResource *s;
    int shape[16], ndim;
    mlx_dtype dtype;
    mlx_array key;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&lower) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&upper) ||
        !get_int_list(env, argv[2], shape, &ndim) ||
        !dtype_from_atom(env, argv[3], &dtype) ||
        !get_optional_key(env, argv[4], &key) ||
        !enif_get_resource(env, argv[5], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_random_truncated_normal(&result, lower->inner, upper->inner, shape, ndim, dtype, key, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// random_categorical(logits, axis, key_or_nil, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_random_categorical(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *logits;
    MlxStreamResource *s;
    int axis;
    mlx_array key;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&logits) ||
        !enif_get_int(env, argv[1], &axis) ||
        !get_optional_key(env, argv[2], &key) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_random_categorical(&result, logits->inner, axis, key, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// random_gumbel(shape, dtype, key_or_nil, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_random_gumbel(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxStreamResource *s;
    int shape[16], ndim;
    mlx_dtype dtype;
    mlx_array key;
    if (!get_int_list(env, argv[0], shape, &ndim) ||
        !dtype_from_atom(env, argv[1], &dtype) ||
        !get_optional_key(env, argv[2], &key) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_random_gumbel(&result, shape, ndim, dtype, key, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// random_laplace(shape, dtype, loc, scale, key_or_nil, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_random_laplace(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxStreamResource *s;
    int shape[16], ndim;
    mlx_dtype dtype;
    double loc, scale;
    mlx_array key;
    if (!get_int_list(env, argv[0], shape, &ndim) ||
        !dtype_from_atom(env, argv[1], &dtype) ||
        !enif_get_double(env, argv[2], &loc) ||
        !enif_get_double(env, argv[3], &scale) ||
        !get_optional_key(env, argv[4], &key) ||
        !enif_get_resource(env, argv[5], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_random_laplace(&result, shape, ndim, dtype, (float)loc, (float)scale, key, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ============================================================
// Wave 3: Linear Algebra ops
// ============================================================

// linalg_inv(a, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_linalg_inv(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_linalg_inv(&result, a->inner, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// linalg_pinv(a, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_linalg_pinv(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_linalg_pinv(&result, a->inner, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// linalg_cholesky(a, upper, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_linalg_cholesky(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "bad argument");
    bool upper = (enif_compare(argv[1], ATOM_TRUE) == 0);
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_linalg_cholesky(&result, a->inner, upper, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// linalg_cholesky_inv(a, upper, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_linalg_cholesky_inv(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "bad argument");
    bool upper = (enif_compare(argv[1], ATOM_TRUE) == 0);
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_linalg_cholesky_inv(&result, a->inner, upper, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// linalg_qr(a, stream) -> {:ok, {q, r}}
static ERL_NIF_TERM nif_linalg_qr(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array q = mlx_array_new();
    mlx_array r = mlx_array_new();
    int ret = mlx_linalg_qr(&q, &r, a->inner, s->inner);
    if (ret != 0) {
        mlx_array_free(q);
        mlx_array_free(r);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    MlxArrayResource *qr = enif_alloc_resource(MLX_ARRAY_RESOURCE, sizeof(MlxArrayResource));
    qr->inner = q;
    ERL_NIF_TERM tq = enif_make_resource(env, qr);
    enif_release_resource(qr);

    MlxArrayResource *rr = enif_alloc_resource(MLX_ARRAY_RESOURCE, sizeof(MlxArrayResource));
    rr->inner = r;
    ERL_NIF_TERM tr = enif_make_resource(env, rr);
    enif_release_resource(rr);

    return MLX_NIF_OK(env, enif_make_tuple2(env, tq, tr));
}

// linalg_svd(a, stream) -> {:ok, [u, s, vt]}
static ERL_NIF_TERM nif_linalg_svd(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_vector_array parts = mlx_vector_array_new();
    int ret = mlx_linalg_svd(&parts, a->inner, true, s->inner);
    if (ret != 0) {
        mlx_vector_array_free(parts);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    size_t n = mlx_vector_array_size(parts);
    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int i = (int)n - 1; i >= 0; i--) {
        mlx_array part = mlx_array_new();
        int get_ret = mlx_vector_array_get(&part, parts, i);
        if (get_ret != 0) {
            mlx_array_free(part);
            mlx_vector_array_free(parts);
            return MLX_NIF_ERROR(env, last_error_msg);
        }
        MlxArrayResource *res = enif_alloc_resource(MLX_ARRAY_RESOURCE, sizeof(MlxArrayResource));
        res->inner = part;
        ERL_NIF_TERM term = enif_make_resource(env, res);
        enif_release_resource(res);
        list = enif_make_list_cell(env, term, list);
    }

    mlx_vector_array_free(parts);
    return MLX_NIF_OK(env, list);
}

// linalg_eigh(a, uplo, stream) -> {:ok, {eigenvalues, eigenvectors}}
static ERL_NIF_TERM nif_linalg_eigh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "bad argument");

    char uplo[2] = "L";
    unsigned int uplo_len;
    if (enif_get_atom_length(env, argv[1], &uplo_len, ERL_NIF_LATIN1) && uplo_len <= 1) {
        enif_get_atom(env, argv[1], uplo, sizeof(uplo), ERL_NIF_LATIN1);
    }

    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array eigenvalues = mlx_array_new();
    mlx_array eigenvectors = mlx_array_new();
    int ret = mlx_linalg_eigh(&eigenvalues, &eigenvectors, a->inner, uplo, s->inner);
    if (ret != 0) {
        mlx_array_free(eigenvalues);
        mlx_array_free(eigenvectors);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    MlxArrayResource *ev = enif_alloc_resource(MLX_ARRAY_RESOURCE, sizeof(MlxArrayResource));
    ev->inner = eigenvalues;
    ERL_NIF_TERM tev = enif_make_resource(env, ev);
    enif_release_resource(ev);

    MlxArrayResource *evec = enif_alloc_resource(MLX_ARRAY_RESOURCE, sizeof(MlxArrayResource));
    evec->inner = eigenvectors;
    ERL_NIF_TERM tevec = enif_make_resource(env, evec);
    enif_release_resource(evec);

    return MLX_NIF_OK(env, enif_make_tuple2(env, tev, tevec));
}

// linalg_eigvalsh(a, uplo, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_linalg_eigvalsh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "bad argument");

    char uplo[2] = "L";
    unsigned int uplo_len;
    if (enif_get_atom_length(env, argv[1], &uplo_len, ERL_NIF_LATIN1) && uplo_len <= 1) {
        enif_get_atom(env, argv[1], uplo, sizeof(uplo), ERL_NIF_LATIN1);
    }

    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_linalg_eigvalsh(&result, a->inner, uplo, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// linalg_lu(a, stream) -> {:ok, [p, l, u]}
static ERL_NIF_TERM nif_linalg_lu(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_vector_array parts = mlx_vector_array_new();
    int ret = mlx_linalg_lu(&parts, a->inner, s->inner);
    if (ret != 0) {
        mlx_vector_array_free(parts);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    size_t n = mlx_vector_array_size(parts);
    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int i = (int)n - 1; i >= 0; i--) {
        mlx_array part = mlx_array_new();
        int get_ret = mlx_vector_array_get(&part, parts, i);
        if (get_ret != 0) {
            mlx_array_free(part);
            mlx_vector_array_free(parts);
            return MLX_NIF_ERROR(env, last_error_msg);
        }
        MlxArrayResource *res = enif_alloc_resource(MLX_ARRAY_RESOURCE, sizeof(MlxArrayResource));
        res->inner = part;
        ERL_NIF_TERM term = enif_make_resource(env, res);
        enif_release_resource(res);
        list = enif_make_list_cell(env, term, list);
    }

    mlx_vector_array_free(parts);
    return MLX_NIF_OK(env, list);
}

// linalg_lu_factor(a, stream) -> {:ok, {lu, pivots}}
static ERL_NIF_TERM nif_linalg_lu_factor(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array lu = mlx_array_new();
    mlx_array pivots = mlx_array_new();
    int ret = mlx_linalg_lu_factor(&lu, &pivots, a->inner, s->inner);
    if (ret != 0) {
        mlx_array_free(lu);
        mlx_array_free(pivots);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    MlxArrayResource *lr = enif_alloc_resource(MLX_ARRAY_RESOURCE, sizeof(MlxArrayResource));
    lr->inner = lu;
    ERL_NIF_TERM tlu = enif_make_resource(env, lr);
    enif_release_resource(lr);

    MlxArrayResource *pr = enif_alloc_resource(MLX_ARRAY_RESOURCE, sizeof(MlxArrayResource));
    pr->inner = pivots;
    ERL_NIF_TERM tp = enif_make_resource(env, pr);
    enif_release_resource(pr);

    return MLX_NIF_OK(env, enif_make_tuple2(env, tlu, tp));
}

// linalg_solve(a, b, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_linalg_solve(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *b;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&b) ||
        !enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_linalg_solve(&result, a->inner, b->inner, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// linalg_solve_triangular(a, b, upper, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_linalg_solve_triangular(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *b;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&b))
        return MLX_NIF_ERROR(env, "bad argument");
    bool upper = (enif_compare(argv[2], ATOM_TRUE) == 0);
    if (!enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_linalg_solve_triangular(&result, a->inner, b->inner, upper, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// linalg_cross(a, b, axis, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_linalg_cross(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a, *b;
    MlxStreamResource *s;
    int axis;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&b) ||
        !enif_get_int(env, argv[2], &axis) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_linalg_cross(&result, a->inner, b->inner, axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// linalg_tri_inv(a, upper, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_linalg_tri_inv(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "bad argument");
    bool upper = (enif_compare(argv[1], ATOM_TRUE) == 0);
    if (!enif_get_resource(env, argv[2], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_linalg_tri_inv(&result, a->inner, upper, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// linalg_norm(a, axes, keepdims, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_linalg_norm(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "bad argument");

    // axes: list of ints or nil (for all axes)
    int axes[16], naxes = 0;
    int *axes_ptr = NULL;
    if (enif_compare(argv[1], ATOM_NIL) != 0) {
        if (!get_int_list(env, argv[1], axes, &naxes))
            return MLX_NIF_ERROR(env, "expected axes list or nil");
        axes_ptr = axes;
    }

    bool keepdims = (enif_compare(argv[2], ATOM_TRUE) == 0);
    if (!enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_linalg_norm_l2(&result, a->inner, axes_ptr, (size_t)naxes, keepdims, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// linalg_norm_p(a, ord, axes, keepdims, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_linalg_norm_p(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    MlxStreamResource *s;
    double ord;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_double(env, argv[1], &ord))
        return MLX_NIF_ERROR(env, "bad argument");

    int axes[16], naxes = 0;
    int *axes_ptr = NULL;
    if (enif_compare(argv[2], ATOM_NIL) != 0) {
        if (!get_int_list(env, argv[2], axes, &naxes))
            return MLX_NIF_ERROR(env, "expected axes list or nil");
        axes_ptr = axes;
    }

    bool keepdims = (enif_compare(argv[3], ATOM_TRUE) == 0);
    if (!enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "expected stream");

    mlx_array result = mlx_array_new();
    int ret = mlx_linalg_norm(&result, a->inner, ord, axes_ptr, (size_t)naxes, keepdims, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ============================================================
// NIF: Wave 4 - FFT operations (10 new NIFs)
// ============================================================

// rfft(arr, n, axis, stream) -> {:ok, arr} — same sig as fft
static ERL_NIF_TERM nif_rfft(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    int n, axis;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &n) ||
        !enif_get_int(env, argv[2], &axis) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_fft_rfft(&result, a->inner, n, axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

static ERL_NIF_TERM nif_irfft(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a; MlxStreamResource *s;
    int n, axis;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) ||
        !enif_get_int(env, argv[1], &n) ||
        !enif_get_int(env, argv[2], &axis) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");
    mlx_array result = mlx_array_new();
    int ret = mlx_fft_irfft(&result, a->inner, n, axis, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// Multi-dim FFT helper: parse (arr, n_list, axes_list, stream)
// n_list and axes_list are Erlang lists of ints
#define DEF_FFT_MULTI(name, mlx_func) \
static ERL_NIF_TERM name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) { \
    (void)argc; \
    MlxArrayResource *a; MlxStreamResource *s; \
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&a) || \
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s)) \
        return MLX_NIF_ERROR(env, "bad argument"); \
    unsigned n_len, axes_len; \
    if (!enif_get_list_length(env, argv[1], &n_len) || \
        !enif_get_list_length(env, argv[2], &axes_len)) \
        return MLX_NIF_ERROR(env, "bad argument"); \
    int n_arr[16], axes_arr[16]; \
    ERL_NIF_TERM head, tail; \
    tail = argv[1]; \
    for (unsigned i = 0; i < n_len && i < 16; i++) { \
        enif_get_list_cell(env, tail, &head, &tail); \
        enif_get_int(env, head, &n_arr[i]); \
    } \
    tail = argv[2]; \
    for (unsigned i = 0; i < axes_len && i < 16; i++) { \
        enif_get_list_cell(env, tail, &head, &tail); \
        enif_get_int(env, head, &axes_arr[i]); \
    } \
    mlx_array result = mlx_array_new(); \
    int ret = mlx_func(&result, a->inner, n_arr, n_len, axes_arr, axes_len, s->inner); \
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg); \
    return wrap_array(env, result); \
}

DEF_FFT_MULTI(nif_fft2,    mlx_fft_fft2)
DEF_FFT_MULTI(nif_fftn,    mlx_fft_fftn)
DEF_FFT_MULTI(nif_ifft2,   mlx_fft_ifft2)
DEF_FFT_MULTI(nif_ifftn,   mlx_fft_ifftn)
DEF_FFT_MULTI(nif_rfft2,   mlx_fft_rfft2)
DEF_FFT_MULTI(nif_rfftn,   mlx_fft_rfftn)
DEF_FFT_MULTI(nif_irfft2,  mlx_fft_irfft2)
DEF_FFT_MULTI(nif_irfftn,  mlx_fft_irfftn)

// ============================================================
// NIF: Wave 5 - Quantization (3 NIFs)
// ============================================================

// quantize(w, group_size, bits, stream) -> {:ok, {quantized, scales, biases}}
static ERL_NIF_TERM nif_quantize(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *w; MlxStreamResource *s;
    int group_size, bits;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&w) ||
        !enif_get_int(env, argv[1], &group_size) ||
        !enif_get_int(env, argv[2], &bits) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_optional_int opt_group_size = {group_size, true};
    mlx_optional_int opt_bits = {bits, true};

    mlx_vector_array parts = mlx_vector_array_new();
    int ret = mlx_quantize(&parts, w->inner, opt_group_size, opt_bits, "affine", s->inner);
    if (ret != 0) {
        mlx_vector_array_free(parts);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    // Extract 3 arrays: quantized, scales, biases
    mlx_array res0 = mlx_array_new(), res1 = mlx_array_new(), res2 = mlx_array_new();
    mlx_vector_array_get(&res0, parts, 0);
    mlx_vector_array_get(&res1, parts, 1);
    mlx_vector_array_get(&res2, parts, 2);
    mlx_vector_array_free(parts);

    ERL_NIF_TERM t0 = wrap_array_raw(env, res0);
    ERL_NIF_TERM t1 = wrap_array_raw(env, res1);
    ERL_NIF_TERM t2 = wrap_array_raw(env, res2);
    return MLX_NIF_OK(env, enif_make_tuple3(env, t0, t1, t2));
}

// dequantize(w, scales, biases, group_size, bits, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_dequantize(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *w, *scales, *biases;
    MlxStreamResource *s;
    int group_size, bits;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&w) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&scales) ||
        !enif_get_resource(env, argv[2], MLX_ARRAY_RESOURCE, (void**)&biases) ||
        !enif_get_int(env, argv[3], &group_size) ||
        !enif_get_int(env, argv[4], &bits) ||
        !enif_get_resource(env, argv[5], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_optional_int opt_group_size = {group_size, true};
    mlx_optional_int opt_bits = {bits, true};
    mlx_optional_dtype opt_dtype = {0, false};

    mlx_array result = mlx_array_new();
    int ret = mlx_dequantize(&result, w->inner, scales->inner, biases->inner,
                              opt_group_size, opt_bits, "affine", opt_dtype, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// quantized_matmul(x, w, scales, biases, transpose, group_size, bits, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_quantized_matmul(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *x, *w, *scales, *biases;
    MlxStreamResource *s;
    int group_size, bits;
    char transpose_str[16];
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&x) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&w) ||
        !enif_get_resource(env, argv[2], MLX_ARRAY_RESOURCE, (void**)&scales) ||
        !enif_get_resource(env, argv[3], MLX_ARRAY_RESOURCE, (void**)&biases) ||
        !enif_get_atom(env, argv[4], transpose_str, sizeof(transpose_str), ERL_NIF_LATIN1) ||
        !enif_get_int(env, argv[5], &group_size) ||
        !enif_get_int(env, argv[6], &bits) ||
        !enif_get_resource(env, argv[7], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    bool transpose = (strcmp(transpose_str, "true") == 0);
    mlx_optional_int opt_group_size = {group_size, true};
    mlx_optional_int opt_bits = {bits, true};

    mlx_array result = mlx_array_new();
    int ret = mlx_quantized_matmul(&result, x->inner, w->inner, scales->inner, biases->inner,
                                    transpose, opt_group_size, opt_bits, "affine", s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ============================================================
// NIF: Wave 6 - Fast ops (4 NIFs)
// ============================================================

// fast_layer_norm(x, weight_or_nil, bias_or_nil, eps, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_fast_layer_norm(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *x; MlxStreamResource *s;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&x) ||
        !enif_get_resource(env, argv[4], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    // weight and bias may be :nil (atom) or array resource
    mlx_array weight = mlx_array_new();
    mlx_array bias = mlx_array_new();
    MlxArrayResource *w_res, *b_res;
    if (enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&w_res))
        weight = w_res->inner;
    if (enif_get_resource(env, argv[2], MLX_ARRAY_RESOURCE, (void**)&b_res))
        bias = b_res->inner;

    double eps;
    if (!enif_get_double(env, argv[3], &eps))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_fast_layer_norm(&result, x->inner, weight, bias, (float)eps, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// fast_rms_norm(x, weight, eps, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_fast_rms_norm(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *x, *w; MlxStreamResource *s;
    double eps;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&x) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&w) ||
        !enif_get_double(env, argv[2], &eps) ||
        !enif_get_resource(env, argv[3], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_fast_rms_norm(&result, x->inner, w->inner, (float)eps, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// fast_rope(x, dims, traditional, base, scale, offset, freqs_or_nil, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_fast_rope(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *x; MlxStreamResource *s;
    int dims, offset;
    double scale_d;
    char trad_str[16];
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&x) ||
        !enif_get_int(env, argv[1], &dims) ||
        !enif_get_atom(env, argv[2], trad_str, sizeof(trad_str), ERL_NIF_LATIN1) ||
        !enif_get_double(env, argv[4], &scale_d) ||
        !enif_get_int(env, argv[5], &offset) ||
        !enif_get_resource(env, argv[7], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    bool traditional = (strcmp(trad_str, "true") == 0);

    // base is optional float
    mlx_optional_float base;
    double base_d;
    if (enif_get_double(env, argv[3], &base_d)) {
        base.value = (float)base_d;
        base.has_value = true;
    } else {
        base.has_value = false;
    }

    // freqs may be :nil or array
    mlx_array freqs = mlx_array_new();
    MlxArrayResource *f_res;
    if (enif_get_resource(env, argv[6], MLX_ARRAY_RESOURCE, (void**)&f_res))
        freqs = f_res->inner;

    mlx_array result = mlx_array_new();
    int ret = mlx_fast_rope(&result, x->inner, dims, traditional, base, (float)scale_d, offset, freqs, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// fast_scaled_dot_product_attention(q, k, v, scale, mask_or_nil, stream) -> {:ok, arr}
// v0.5.0: added mask_mode string, sinks array; removed memory_efficient_threshold
static ERL_NIF_TERM nif_fast_sdpa(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *q, *k, *v; MlxStreamResource *s;
    double scale_d;
    if (!enif_get_resource(env, argv[0], MLX_ARRAY_RESOURCE, (void**)&q) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&k) ||
        !enif_get_resource(env, argv[2], MLX_ARRAY_RESOURCE, (void**)&v) ||
        !enif_get_double(env, argv[3], &scale_d) ||
        !enif_get_resource(env, argv[5], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    // mask may be :nil or array; determine mask_mode accordingly
    mlx_array mask_arr = mlx_array_new();
    const char *mask_mode = "";
    MlxArrayResource *m_res;
    if (enif_get_resource(env, argv[4], MLX_ARRAY_RESOURCE, (void**)&m_res)) {
        mask_arr = m_res->inner;
        mask_mode = "array";
    }

    // sinks not exposed yet — pass null array
    mlx_array sinks = mlx_array_new();

    mlx_array result = mlx_array_new();
    int ret = mlx_fast_scaled_dot_product_attention(&result, q->inner, k->inner, v->inner,
                                                     (float)scale_d, mask_mode, mask_arr, sinks, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// ============================================================
// NIF: Wave 7 - I/O ops (4 NIFs)
// ============================================================

// save(path, arr) -> :ok | {:error, msg}
static ERL_NIF_TERM nif_save(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxArrayResource *a;
    char path[4096];
    if (!enif_get_string(env, argv[0], path, sizeof(path), ERL_NIF_LATIN1) ||
        !enif_get_resource(env, argv[1], MLX_ARRAY_RESOURCE, (void**)&a))
        return MLX_NIF_ERROR(env, "bad argument");

    int ret = mlx_save(path, a->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return ATOM_OK;
}

// load(path, stream) -> {:ok, arr}
static ERL_NIF_TERM nif_load(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxStreamResource *s;
    char path[4096];
    if (!enif_get_string(env, argv[0], path, sizeof(path), ERL_NIF_LATIN1) ||
        !enif_get_resource(env, argv[1], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_array result = mlx_array_new();
    int ret = mlx_load(&result, path, s->inner);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
}

// save_safetensors(path, keys_list, arrays_list, metadata_keys, metadata_vals) -> :ok | {:error, msg}
static ERL_NIF_TERM nif_save_safetensors(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    char path[4096];
    if (!enif_get_string(env, argv[0], path, sizeof(path), ERL_NIF_LATIN1))
        return MLX_NIF_ERROR(env, "bad argument");

    // Build arrays map from keys_list and arrays_list
    mlx_map_string_to_array arr_map = mlx_map_string_to_array_new();
    unsigned keys_len;
    if (!enif_get_list_length(env, argv[1], &keys_len))
        return MLX_NIF_ERROR(env, "bad argument: keys must be a list");

    ERL_NIF_TERM keys_head, keys_tail = argv[1];
    ERL_NIF_TERM arrs_head, arrs_tail = argv[2];
    for (unsigned i = 0; i < keys_len; i++) {
        if (!enif_get_list_cell(env, keys_tail, &keys_head, &keys_tail) ||
            !enif_get_list_cell(env, arrs_tail, &arrs_head, &arrs_tail))
            return MLX_NIF_ERROR(env, "bad argument: list mismatch");

        char key[256];
        if (!enif_get_string(env, keys_head, key, sizeof(key), ERL_NIF_LATIN1))
            return MLX_NIF_ERROR(env, "bad argument: key must be string");
        MlxArrayResource *arr;
        if (!enif_get_resource(env, arrs_head, MLX_ARRAY_RESOURCE, (void**)&arr))
            return MLX_NIF_ERROR(env, "bad argument: value must be array");
        mlx_map_string_to_array_insert(arr_map, key, arr->inner);
    }

    // Build metadata map
    mlx_map_string_to_string meta_map = mlx_map_string_to_string_new();
    unsigned meta_len;
    if (enif_get_list_length(env, argv[3], &meta_len) && meta_len > 0) {
        ERL_NIF_TERM mk_head, mk_tail = argv[3];
        ERL_NIF_TERM mv_head, mv_tail = argv[4];
        for (unsigned i = 0; i < meta_len; i++) {
            if (!enif_get_list_cell(env, mk_tail, &mk_head, &mk_tail) ||
                !enif_get_list_cell(env, mv_tail, &mv_head, &mv_tail))
                break;
            char mk[256], mv[1024];
            if (enif_get_string(env, mk_head, mk, sizeof(mk), ERL_NIF_LATIN1) &&
                enif_get_string(env, mv_head, mv, sizeof(mv), ERL_NIF_LATIN1))
                mlx_map_string_to_string_insert(meta_map, mk, mv);
        }
    }

    int ret = mlx_save_safetensors(path, arr_map, meta_map);
    mlx_map_string_to_array_free(arr_map);
    mlx_map_string_to_string_free(meta_map);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return ATOM_OK;
}

// load_safetensors(path, stream) -> {:ok, {[{key, arr}], [{meta_key, meta_val}]}}
static ERL_NIF_TERM nif_load_safetensors(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    MlxStreamResource *s;
    char path[4096];
    if (!enif_get_string(env, argv[0], path, sizeof(path), ERL_NIF_LATIN1) ||
        !enif_get_resource(env, argv[1], MLX_STREAM_RESOURCE, (void**)&s))
        return MLX_NIF_ERROR(env, "bad argument");

    mlx_map_string_to_array arr_map = mlx_map_string_to_array_new();
    mlx_map_string_to_string meta_map = mlx_map_string_to_string_new();
    int ret = mlx_load_safetensors(&arr_map, &meta_map, path, s->inner);
    if (ret != 0) {
        mlx_map_string_to_array_free(arr_map);
        mlx_map_string_to_string_free(meta_map);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    // Build arrays list: [{key, arr_resource}, ...]
    ERL_NIF_TERM arr_list = enif_make_list(env, 0);
    mlx_map_string_to_array_iterator arr_it = mlx_map_string_to_array_iterator_new(arr_map);
    const char* key;
    mlx_array val = mlx_array_new();
    while (mlx_map_string_to_array_iterator_next(&key, &val, arr_it) == 0 && key != NULL) {
        // Make a copy of the array since the iterator's reference might be invalidated
        mlx_array copy = mlx_array_new();
        mlx_array_set(&copy, val);
        ERL_NIF_TERM key_term = enif_make_string(env, key, ERL_NIF_LATIN1);
        ERL_NIF_TERM val_term = wrap_array_raw(env, copy);
        ERL_NIF_TERM pair = enif_make_tuple2(env, key_term, val_term);
        arr_list = enif_make_list_cell(env, pair, arr_list);
    }
    mlx_map_string_to_array_iterator_free(arr_it);
    mlx_array_free(val);

    // Build metadata list: [{key, val}, ...]
    ERL_NIF_TERM meta_list = enif_make_list(env, 0);
    mlx_map_string_to_string_iterator meta_it = mlx_map_string_to_string_iterator_new(meta_map);
    const char* mk;
    const char* mv;
    while (mlx_map_string_to_string_iterator_next(&mk, &mv, meta_it) == 0 && mk != NULL) {
        ERL_NIF_TERM mk_term = enif_make_string(env, mk, ERL_NIF_LATIN1);
        ERL_NIF_TERM mv_term = enif_make_string(env, mv, ERL_NIF_LATIN1);
        ERL_NIF_TERM pair = enif_make_tuple2(env, mk_term, mv_term);
        meta_list = enif_make_list_cell(env, pair, meta_list);
    }
    mlx_map_string_to_string_iterator_free(meta_it);

    mlx_map_string_to_array_free(arr_map);
    mlx_map_string_to_string_free(meta_map);

    return MLX_NIF_OK(env, enif_make_tuple2(env, arr_list, meta_list));
}

// ============================================================
// Closure Bridge: Trampoline pattern for Elixir ↔ MLX transforms
// ============================================================

ErlNifResourceType *MLX_CLOSURE_BRIDGE_RESOURCE = NULL;

static void closure_bridge_destructor(ErlNifEnv* env, void* obj) {
    (void)env;
    ClosureBridgePayload *bridge = (ClosureBridgePayload*)obj;
    if (bridge->mutex) {
        enif_mutex_destroy(bridge->mutex);
        bridge->mutex = NULL;
    }
    if (bridge->cond) {
        enif_cond_destroy(bridge->cond);
        bridge->cond = NULL;
    }
    if (bridge->msg_env) {
        enif_free_env(bridge->msg_env);
        bridge->msg_env = NULL;
    }
    if (bridge->result.ctx) {
        mlx_vector_array_free(bridge->result);
        bridge->result.ctx = NULL;
    }
}

// Trampoline function called by MLX during transforms (e.g. value_and_grad).
// Sends inputs to the Elixir helper process, then blocks waiting for results.
static int closure_trampoline(mlx_vector_array *out, const mlx_vector_array in, void *payload) {
    ClosureBridgePayload *bridge = (ClosureBridgePayload *)payload;

    // Build list of array refs to send to the helper process
    size_t n = mlx_vector_array_size(in);
    ErlNifEnv *msg_env = bridge->msg_env;
    enif_clear_env(msg_env);

    // Build a list of array resource terms for the inputs
    ERL_NIF_TERM list = enif_make_list(msg_env, 0);
    // Build in reverse order, then the list will be reversed
    for (int i = (int)n - 1; i >= 0; i--) {
        mlx_array elem = mlx_array_new();
        if (mlx_vector_array_get(&elem, in, (size_t)i) != 0) {
            mlx_array_free(elem);
            return 1;
        }
        // Copy the array so it persists after the vector is freed
        mlx_array copy = mlx_array_new();
        mlx_array_set(&copy, elem);
        mlx_array_free(elem);

        MlxArrayResource *res = enif_alloc_resource(MLX_ARRAY_RESOURCE, sizeof(MlxArrayResource));
        res->inner = copy;
        ERL_NIF_TERM term = enif_make_resource(msg_env, res);
        enif_release_resource(res);
        list = enif_make_list_cell(msg_env, term, list);
    }

    // Send {:trampoline_call, bridge_ref, [array_refs...]} to helper
    ERL_NIF_TERM bridge_ref = enif_make_resource(msg_env, bridge);
    ERL_NIF_TERM atom_call = enif_make_atom(msg_env, "trampoline_call");
    ERL_NIF_TERM msg = enif_make_tuple3(msg_env, atom_call, bridge_ref, list);

    if (!enif_send(NULL, &bridge->helper_pid, msg_env, msg)) {
        return 1;
    }

    // Block until the helper responds
    enif_mutex_lock(bridge->mutex);
    while (!bridge->ready && !bridge->error) {
        enif_cond_wait(bridge->cond, bridge->mutex);
    }
    int err = bridge->error;
    bridge->ready = 0;

    if (!err && bridge->result.ctx) {
        mlx_vector_array_set(out, bridge->result);
    }
    enif_mutex_unlock(bridge->mutex);

    return err;
}

// Destructor for the closure payload (called by mlx_closure_free)
static void closure_payload_dtor(void *payload) {
    // The bridge resource is ref-counted by Erlang; don't free it here.
    // The ClosureBridgePayload destructor handles cleanup when GC'd.
    (void)payload;
}

// NIF: closure_respond(bridge_ref, array_ref_list)
// Called by the helper process to return results to the blocked trampoline.
static ERL_NIF_TERM nif_closure_respond(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    ClosureBridgePayload *bridge;
    if (!enif_get_resource(env, argv[0], MLX_CLOSURE_BRIDGE_RESOURCE, (void**)&bridge))
        return enif_make_badarg(env);

    // Parse list of array refs into a vector_array
    unsigned int len;
    if (!enif_get_list_length(env, argv[1], &len))
        return enif_make_badarg(env);

    mlx_vector_array result = mlx_vector_array_new();
    if (!result.ctx) return MLX_NIF_ERROR(env, "failed to create vector_array");

    ERL_NIF_TERM head, tail = argv[1];
    for (unsigned int i = 0; i < len; i++) {
        if (!enif_get_list_cell(env, tail, &head, &tail)) {
            mlx_vector_array_free(result);
            return enif_make_badarg(env);
        }
        MlxArrayResource *arr;
        if (!enif_get_resource(env, head, MLX_ARRAY_RESOURCE, (void**)&arr)) {
            mlx_vector_array_free(result);
            return enif_make_badarg(env);
        }
        mlx_vector_array_append_value(result, arr->inner);
    }

    // Signal the trampoline
    enif_mutex_lock(bridge->mutex);
    if (bridge->result.ctx) mlx_vector_array_free(bridge->result);
    bridge->result = result;
    bridge->ready = 1;
    bridge->error = 0;
    enif_cond_signal(bridge->cond);
    enif_mutex_unlock(bridge->mutex);

    return ATOM_OK;
}

// NIF: closure_respond_error(bridge_ref)
// Called by the helper process to signal an error to the blocked trampoline.
static ERL_NIF_TERM nif_closure_respond_error(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    ClosureBridgePayload *bridge;
    if (!enif_get_resource(env, argv[0], MLX_CLOSURE_BRIDGE_RESOURCE, (void**)&bridge))
        return enif_make_badarg(env);

    enif_mutex_lock(bridge->mutex);
    bridge->error = 1;
    bridge->ready = 0;
    enif_cond_signal(bridge->cond);
    enif_mutex_unlock(bridge->mutex);

    return ATOM_OK;
}

// NIF: value_and_grad_apply(helper_pid, array_ref_list, argnums_list)
// Runs on dirty scheduler. Creates closure bridge, wraps Elixir function,
// calls mlx_value_and_grad, then mlx_closure_value_and_grad_apply.
static ERL_NIF_TERM nif_value_and_grad_apply(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;

    // argv[0] = helper_pid, argv[1] = input array_ref list, argv[2] = argnums list
    ErlNifPid helper_pid;
    if (!enif_get_local_pid(env, argv[0], &helper_pid))
        return MLX_NIF_ERROR(env, "expected helper pid");

    // Parse input arrays
    unsigned int input_len;
    if (!enif_get_list_length(env, argv[1], &input_len))
        return enif_make_badarg(env);

    mlx_vector_array inputs = mlx_vector_array_new();
    if (!inputs.ctx) return MLX_NIF_ERROR(env, "failed to create input vector");

    ERL_NIF_TERM head, tail = argv[1];
    for (unsigned int i = 0; i < input_len; i++) {
        if (!enif_get_list_cell(env, tail, &head, &tail)) {
            mlx_vector_array_free(inputs);
            return enif_make_badarg(env);
        }
        MlxArrayResource *arr;
        if (!enif_get_resource(env, head, MLX_ARRAY_RESOURCE, (void**)&arr)) {
            mlx_vector_array_free(inputs);
            return enif_make_badarg(env);
        }
        mlx_vector_array_append_value(inputs, arr->inner);
    }

    // Parse argnums
    unsigned int argnums_len;
    if (!enif_get_list_length(env, argv[2], &argnums_len))  {
        mlx_vector_array_free(inputs);
        return enif_make_badarg(env);
    }

    int *argnums = (int*)enif_alloc(argnums_len * sizeof(int));
    tail = argv[2];
    for (unsigned int i = 0; i < argnums_len; i++) {
        if (!enif_get_list_cell(env, tail, &head, &tail) || !enif_get_int(env, head, &argnums[i])) {
            enif_free(argnums);
            mlx_vector_array_free(inputs);
            return enif_make_badarg(env);
        }
    }

    // Create closure bridge resource
    ClosureBridgePayload *bridge = enif_alloc_resource(
        MLX_CLOSURE_BRIDGE_RESOURCE, sizeof(ClosureBridgePayload));
    memset(bridge, 0, sizeof(ClosureBridgePayload));
    bridge->msg_env = enif_alloc_env();
    bridge->helper_pid = helper_pid;
    bridge->mutex = enif_mutex_create("closure_bridge");
    bridge->cond = enif_cond_create("closure_bridge");
    bridge->result.ctx = NULL;
    bridge->ready = 0;
    bridge->error = 0;

    // Keep a reference to bridge so it won't be GC'd during the call
    enif_keep_resource(bridge);

    // Create MLX closure with our trampoline
    mlx_closure closure = mlx_closure_new_func_payload(
        closure_trampoline, (void*)bridge, closure_payload_dtor);

    if (!closure.ctx) {
        enif_release_resource(bridge);
        enif_release_resource(bridge);
        enif_free(argnums);
        mlx_vector_array_free(inputs);
        return MLX_NIF_ERROR(env, "failed to create closure");
    }

    // Create value_and_grad transform
    mlx_closure_value_and_grad vag = mlx_closure_value_and_grad_new();
    int ret = mlx_value_and_grad(&vag, closure, argnums, argnums_len);
    enif_free(argnums);

    if (ret != 0) {
        mlx_closure_value_and_grad_free(vag);
        mlx_closure_free(closure);
        enif_release_resource(bridge);
        enif_release_resource(bridge);
        mlx_vector_array_free(inputs);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    // Apply the transform
    mlx_vector_array values = mlx_vector_array_new();
    mlx_vector_array grads = mlx_vector_array_new();
    ret = mlx_closure_value_and_grad_apply(&values, &grads, vag, inputs);

    mlx_closure_value_and_grad_free(vag);
    mlx_closure_free(closure);
    enif_release_resource(bridge); // release the keep
    enif_release_resource(bridge); // release the alloc
    mlx_vector_array_free(inputs);

    if (ret != 0) {
        mlx_vector_array_free(values);
        mlx_vector_array_free(grads);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    // Convert values and grads to Erlang lists of array refs
    size_t nv = mlx_vector_array_size(values);
    size_t ng = mlx_vector_array_size(grads);

    ERL_NIF_TERM values_list = enif_make_list(env, 0);
    for (int i = (int)nv - 1; i >= 0; i--) {
        mlx_array elem = mlx_array_new();
        mlx_vector_array_get(&elem, values, (size_t)i);
        mlx_array copy = mlx_array_new();
        mlx_array_set(&copy, elem);
        mlx_array_free(elem);
        values_list = enif_make_list_cell(env, wrap_array_raw(env, copy), values_list);
    }

    ERL_NIF_TERM grads_list = enif_make_list(env, 0);
    for (int i = (int)ng - 1; i >= 0; i--) {
        mlx_array elem = mlx_array_new();
        mlx_vector_array_get(&elem, grads, (size_t)i);
        mlx_array copy = mlx_array_new();
        mlx_array_set(&copy, elem);
        mlx_array_free(elem);
        grads_list = enif_make_list_cell(env, wrap_array_raw(env, copy), grads_list);
    }

    mlx_vector_array_free(values);
    mlx_vector_array_free(grads);

    return MLX_NIF_OK(env, enif_make_tuple2(env, values_list, grads_list));
}

// NIF: vjp_apply(helper_pid, primals_list, cotangents_list)
static ERL_NIF_TERM nif_vjp_apply(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;

    ErlNifPid helper_pid;
    if (!enif_get_local_pid(env, argv[0], &helper_pid))
        return MLX_NIF_ERROR(env, "expected helper pid");

    // Parse primals
    unsigned int plen;
    if (!enif_get_list_length(env, argv[1], &plen))
        return enif_make_badarg(env);

    mlx_vector_array primals = mlx_vector_array_new();
    ERL_NIF_TERM head, tail = argv[1];
    for (unsigned int i = 0; i < plen; i++) {
        enif_get_list_cell(env, tail, &head, &tail);
        MlxArrayResource *arr;
        if (!enif_get_resource(env, head, MLX_ARRAY_RESOURCE, (void**)&arr)) {
            mlx_vector_array_free(primals);
            return enif_make_badarg(env);
        }
        mlx_vector_array_append_value(primals, arr->inner);
    }

    // Parse cotangents
    unsigned int clen;
    if (!enif_get_list_length(env, argv[2], &clen))  {
        mlx_vector_array_free(primals);
        return enif_make_badarg(env);
    }

    mlx_vector_array cotangents = mlx_vector_array_new();
    tail = argv[2];
    for (unsigned int i = 0; i < clen; i++) {
        enif_get_list_cell(env, tail, &head, &tail);
        MlxArrayResource *arr;
        if (!enif_get_resource(env, head, MLX_ARRAY_RESOURCE, (void**)&arr)) {
            mlx_vector_array_free(primals);
            mlx_vector_array_free(cotangents);
            return enif_make_badarg(env);
        }
        mlx_vector_array_append_value(cotangents, arr->inner);
    }

    // Create closure bridge
    ClosureBridgePayload *bridge = enif_alloc_resource(
        MLX_CLOSURE_BRIDGE_RESOURCE, sizeof(ClosureBridgePayload));
    memset(bridge, 0, sizeof(ClosureBridgePayload));
    bridge->msg_env = enif_alloc_env();
    bridge->helper_pid = helper_pid;
    bridge->mutex = enif_mutex_create("vjp_bridge");
    bridge->cond = enif_cond_create("vjp_bridge");
    bridge->result.ctx = NULL;
    bridge->ready = 0;
    bridge->error = 0;
    enif_keep_resource(bridge);

    mlx_closure closure = mlx_closure_new_func_payload(
        closure_trampoline, (void*)bridge, closure_payload_dtor);

    if (!closure.ctx) {
        enif_release_resource(bridge);
        enif_release_resource(bridge);
        mlx_vector_array_free(primals);
        mlx_vector_array_free(cotangents);
        return MLX_NIF_ERROR(env, "failed to create closure");
    }

    // Call mlx_vjp
    mlx_vector_array out_primals = mlx_vector_array_new();
    mlx_vector_array out_vjps = mlx_vector_array_new();
    int ret = mlx_vjp(&out_primals, &out_vjps, closure, primals, cotangents);

    mlx_closure_free(closure);
    enif_release_resource(bridge);
    enif_release_resource(bridge);
    mlx_vector_array_free(primals);
    mlx_vector_array_free(cotangents);

    if (ret != 0) {
        mlx_vector_array_free(out_primals);
        mlx_vector_array_free(out_vjps);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    // Convert to Erlang lists
    size_t np = mlx_vector_array_size(out_primals);
    size_t nv = mlx_vector_array_size(out_vjps);

    ERL_NIF_TERM primals_result = enif_make_list(env, 0);
    for (int i = (int)np - 1; i >= 0; i--) {
        mlx_array elem = mlx_array_new();
        mlx_vector_array_get(&elem, out_primals, (size_t)i);
        mlx_array copy = mlx_array_new();
        mlx_array_set(&copy, elem);
        mlx_array_free(elem);
        primals_result = enif_make_list_cell(env, wrap_array_raw(env, copy), primals_result);
    }

    ERL_NIF_TERM vjps_result = enif_make_list(env, 0);
    for (int i = (int)nv - 1; i >= 0; i--) {
        mlx_array elem = mlx_array_new();
        mlx_vector_array_get(&elem, out_vjps, (size_t)i);
        mlx_array copy = mlx_array_new();
        mlx_array_set(&copy, elem);
        mlx_array_free(elem);
        vjps_result = enif_make_list_cell(env, wrap_array_raw(env, copy), vjps_result);
    }

    mlx_vector_array_free(out_primals);
    mlx_vector_array_free(out_vjps);

    return MLX_NIF_OK(env, enif_make_tuple2(env, primals_result, vjps_result));
}

// NIF: jvp_apply(helper_pid, primals_list, tangents_list)
static ERL_NIF_TERM nif_jvp_apply(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;

    ErlNifPid helper_pid;
    if (!enif_get_local_pid(env, argv[0], &helper_pid))
        return MLX_NIF_ERROR(env, "expected helper pid");

    // Parse primals
    unsigned int plen;
    if (!enif_get_list_length(env, argv[1], &plen))
        return enif_make_badarg(env);

    mlx_vector_array primals = mlx_vector_array_new();
    ERL_NIF_TERM head, tail = argv[1];
    for (unsigned int i = 0; i < plen; i++) {
        enif_get_list_cell(env, tail, &head, &tail);
        MlxArrayResource *arr;
        if (!enif_get_resource(env, head, MLX_ARRAY_RESOURCE, (void**)&arr)) {
            mlx_vector_array_free(primals);
            return enif_make_badarg(env);
        }
        mlx_vector_array_append_value(primals, arr->inner);
    }

    // Parse tangents
    unsigned int tlen;
    if (!enif_get_list_length(env, argv[2], &tlen)) {
        mlx_vector_array_free(primals);
        return enif_make_badarg(env);
    }

    mlx_vector_array tangents = mlx_vector_array_new();
    tail = argv[2];
    for (unsigned int i = 0; i < tlen; i++) {
        enif_get_list_cell(env, tail, &head, &tail);
        MlxArrayResource *arr;
        if (!enif_get_resource(env, head, MLX_ARRAY_RESOURCE, (void**)&arr)) {
            mlx_vector_array_free(primals);
            mlx_vector_array_free(tangents);
            return enif_make_badarg(env);
        }
        mlx_vector_array_append_value(tangents, arr->inner);
    }

    // Create closure bridge
    ClosureBridgePayload *bridge = enif_alloc_resource(
        MLX_CLOSURE_BRIDGE_RESOURCE, sizeof(ClosureBridgePayload));
    memset(bridge, 0, sizeof(ClosureBridgePayload));
    bridge->msg_env = enif_alloc_env();
    bridge->helper_pid = helper_pid;
    bridge->mutex = enif_mutex_create("jvp_bridge");
    bridge->cond = enif_cond_create("jvp_bridge");
    bridge->result.ctx = NULL;
    bridge->ready = 0;
    bridge->error = 0;
    enif_keep_resource(bridge);

    mlx_closure closure = mlx_closure_new_func_payload(
        closure_trampoline, (void*)bridge, closure_payload_dtor);

    if (!closure.ctx) {
        enif_release_resource(bridge);
        enif_release_resource(bridge);
        mlx_vector_array_free(primals);
        mlx_vector_array_free(tangents);
        return MLX_NIF_ERROR(env, "failed to create closure");
    }

    // Call mlx_jvp
    mlx_vector_array out_primals = mlx_vector_array_new();
    mlx_vector_array out_tangents = mlx_vector_array_new();
    int ret = mlx_jvp(&out_primals, &out_tangents, closure, primals, tangents);

    mlx_closure_free(closure);
    enif_release_resource(bridge);
    enif_release_resource(bridge);
    mlx_vector_array_free(primals);
    mlx_vector_array_free(tangents);

    if (ret != 0) {
        mlx_vector_array_free(out_primals);
        mlx_vector_array_free(out_tangents);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    // Convert to Erlang lists
    size_t np = mlx_vector_array_size(out_primals);
    size_t nt = mlx_vector_array_size(out_tangents);

    ERL_NIF_TERM primals_result = enif_make_list(env, 0);
    for (int i = (int)np - 1; i >= 0; i--) {
        mlx_array elem = mlx_array_new();
        mlx_vector_array_get(&elem, out_primals, (size_t)i);
        mlx_array copy = mlx_array_new();
        mlx_array_set(&copy, elem);
        mlx_array_free(elem);
        primals_result = enif_make_list_cell(env, wrap_array_raw(env, copy), primals_result);
    }

    ERL_NIF_TERM tangents_result = enif_make_list(env, 0);
    for (int i = (int)nt - 1; i >= 0; i--) {
        mlx_array elem = mlx_array_new();
        mlx_vector_array_get(&elem, out_tangents, (size_t)i);
        mlx_array copy = mlx_array_new();
        mlx_array_set(&copy, elem);
        mlx_array_free(elem);
        tangents_result = enif_make_list_cell(env, wrap_array_raw(env, copy), tangents_result);
    }

    mlx_vector_array_free(out_primals);
    mlx_vector_array_free(out_tangents);

    return MLX_NIF_OK(env, enif_make_tuple2(env, primals_result, tangents_result));
}

// --- NIF: vmap_apply ---
// Vectorized mapping via mlx_detail_vmap_trace + mlx_detail_vmap_replace
static ERL_NIF_TERM nif_vmap_apply(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;

    ErlNifPid helper_pid;
    if (!enif_get_local_pid(env, argv[0], &helper_pid))
        return MLX_NIF_ERROR(env, "expected helper pid");

    // Parse inputs (list of array refs)
    unsigned int ilen;
    if (!enif_get_list_length(env, argv[1], &ilen))
        return enif_make_badarg(env);

    mlx_vector_array inputs = mlx_vector_array_new();
    ERL_NIF_TERM head, tail = argv[1];
    for (unsigned int i = 0; i < ilen; i++) {
        enif_get_list_cell(env, tail, &head, &tail);
        MlxArrayResource *arr;
        if (!enif_get_resource(env, head, MLX_ARRAY_RESOURCE, (void**)&arr)) {
            mlx_vector_array_free(inputs);
            return enif_make_badarg(env);
        }
        mlx_vector_array_append_value(inputs, arr->inner);
    }

    // Parse in_axes (list of ints)
    unsigned int in_axes_len;
    if (!enif_get_list_length(env, argv[2], &in_axes_len)) {
        mlx_vector_array_free(inputs);
        return enif_make_badarg(env);
    }
    int *in_axes = enif_alloc(sizeof(int) * in_axes_len);
    tail = argv[2];
    for (unsigned int i = 0; i < in_axes_len; i++) {
        enif_get_list_cell(env, tail, &head, &tail);
        if (!enif_get_int(env, head, &in_axes[i])) {
            enif_free(in_axes);
            mlx_vector_array_free(inputs);
            return enif_make_badarg(env);
        }
    }

    // Parse out_axes (list of ints)
    unsigned int out_axes_len;
    if (!enif_get_list_length(env, argv[3], &out_axes_len)) {
        enif_free(in_axes);
        mlx_vector_array_free(inputs);
        return enif_make_badarg(env);
    }
    int *out_axes = enif_alloc(sizeof(int) * out_axes_len);
    tail = argv[3];
    for (unsigned int i = 0; i < out_axes_len; i++) {
        enif_get_list_cell(env, tail, &head, &tail);
        if (!enif_get_int(env, head, &out_axes[i])) {
            enif_free(in_axes);
            enif_free(out_axes);
            mlx_vector_array_free(inputs);
            return enif_make_badarg(env);
        }
    }

    // Create closure bridge
    ClosureBridgePayload *bridge = enif_alloc_resource(
        MLX_CLOSURE_BRIDGE_RESOURCE, sizeof(ClosureBridgePayload));
    memset(bridge, 0, sizeof(ClosureBridgePayload));
    bridge->msg_env = enif_alloc_env();
    bridge->helper_pid = helper_pid;
    bridge->mutex = enif_mutex_create("vmap_bridge");
    bridge->cond = enif_cond_create("vmap_bridge");
    bridge->result.ctx = NULL;
    bridge->ready = 0;
    bridge->error = 0;
    enif_keep_resource(bridge);

    mlx_closure closure = mlx_closure_new_func_payload(
        closure_trampoline, (void*)bridge, closure_payload_dtor);

    if (!closure.ctx) {
        enif_release_resource(bridge);
        enif_release_resource(bridge);
        enif_free(in_axes);
        enif_free(out_axes);
        mlx_vector_array_free(inputs);
        return MLX_NIF_ERROR(env, "failed to create closure");
    }

    // Step 1: vmap_trace — gets traced inputs and outputs
    mlx_vector_array s_inputs = mlx_vector_array_new();
    mlx_vector_array s_outputs = mlx_vector_array_new();
    int ret = mlx_detail_vmap_trace(
        &s_inputs, &s_outputs, closure, inputs, in_axes, (size_t)in_axes_len);

    mlx_closure_free(closure);
    enif_release_resource(bridge);
    enif_release_resource(bridge);

    if (ret != 0) {
        enif_free(in_axes);
        enif_free(out_axes);
        mlx_vector_array_free(inputs);
        mlx_vector_array_free(s_inputs);
        mlx_vector_array_free(s_outputs);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    // Step 2: vmap_replace — replaces traced arrays with actual vmapped results
    mlx_vector_array result = mlx_vector_array_new();
    ret = mlx_detail_vmap_replace(
        &result, inputs, s_inputs, s_outputs,
        in_axes, (size_t)in_axes_len, out_axes, (size_t)out_axes_len);

    enif_free(in_axes);
    enif_free(out_axes);
    mlx_vector_array_free(inputs);
    mlx_vector_array_free(s_inputs);
    mlx_vector_array_free(s_outputs);

    if (ret != 0) {
        mlx_vector_array_free(result);
        return MLX_NIF_ERROR(env, last_error_msg);
    }

    // Convert results to Erlang list
    size_t nout = mlx_vector_array_size(result);
    ERL_NIF_TERM result_list = enif_make_list(env, 0);
    for (int i = (int)nout - 1; i >= 0; i--) {
        mlx_array elem = mlx_array_new();
        mlx_vector_array_get(&elem, result, (size_t)i);
        mlx_array copy = mlx_array_new();
        mlx_array_set(&copy, elem);
        mlx_array_free(elem);
        result_list = enif_make_list_cell(env, wrap_array_raw(env, copy), result_list);
    }

    mlx_vector_array_free(result);
    return MLX_NIF_OK(env, result_list);
}

// ============================================================
// NIF: Load / Upgrade / Unload
// ============================================================

#define MAKE_ATOM(env, name, var) var = enif_make_atom(env, name)

static int load(ErlNifEnv* env, void** priv, ERL_NIF_TERM load_info) {
    (void)priv; (void)load_info;

    // Register resource types with destructors (RT-3, S1)
    MLX_ARRAY_RESOURCE = enif_open_resource_type(env, NULL, "mlx_array",
        array_destructor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    MLX_STREAM_RESOURCE = enif_open_resource_type(env, NULL, "mlx_stream",
        stream_destructor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    MLX_DEVICE_RESOURCE = enif_open_resource_type(env, NULL, "mlx_device",
        device_destructor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    MLX_VECTOR_ARRAY_RESOURCE = enif_open_resource_type(env, NULL, "mlx_vector_array",
        vector_array_destructor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    MLX_CLOSURE_BRIDGE_RESOURCE = enif_open_resource_type(env, NULL, "mlx_closure_bridge",
        closure_bridge_destructor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);

    if (!MLX_ARRAY_RESOURCE || !MLX_STREAM_RESOURCE || !MLX_DEVICE_RESOURCE ||
        !MLX_VECTOR_ARRAY_RESOURCE || !MLX_CLOSURE_BRIDGE_RESOURCE)
        return -1;

    // Initialize atoms
    MAKE_ATOM(env, "ok",    ATOM_OK);
    MAKE_ATOM(env, "error", ATOM_ERROR);
    MAKE_ATOM(env, "true",  ATOM_TRUE);
    MAKE_ATOM(env, "false", ATOM_FALSE);
    MAKE_ATOM(env, "nil",   ATOM_NIL);

    MAKE_ATOM(env, "bool", ATOM_BOOL);
    MAKE_ATOM(env, "u8",   ATOM_U8);
    MAKE_ATOM(env, "u16",  ATOM_U16);
    MAKE_ATOM(env, "u32",  ATOM_U32);
    MAKE_ATOM(env, "u64",  ATOM_U64);
    MAKE_ATOM(env, "s8",   ATOM_S8);
    MAKE_ATOM(env, "s16",  ATOM_S16);
    MAKE_ATOM(env, "s32",  ATOM_S32);
    MAKE_ATOM(env, "s64",  ATOM_S64);
    MAKE_ATOM(env, "f16",  ATOM_F16);
    MAKE_ATOM(env, "f32",  ATOM_F32);
    MAKE_ATOM(env, "bf16", ATOM_BF16);
    MAKE_ATOM(env, "c64",  ATOM_C64);

    MAKE_ATOM(env, "cpu", ATOM_CPU);
    MAKE_ATOM(env, "gpu", ATOM_GPU);

    // Set error handler (RT-4)
    mlx_set_error_handler(mlx_error_handler, NULL, NULL);

    return 0;
}

static int upgrade(ErlNifEnv* env, void** priv, void** old_priv, ERL_NIF_TERM load_info) {
    return load(env, priv, load_info);
    (void)old_priv;
}

// ============================================================
// NIF function table
// ============================================================

static ErlNifFunc nif_funcs[] = {
    // Array creation
    {"from_binary",        3, nif_from_binary,        0},
    {"zeros",              3, nif_zeros,              0},
    {"ones",               3, nif_ones,               0},
    {"full",               4, nif_full,               0},
    {"eye",                5, nif_eye,                0},
    {"arange",             5, nif_arange,             0},
    {"linspace",           5, nif_linspace,           0},

    // Array properties
    {"shape",              1, nif_shape,              0},
    {"dtype",              1, nif_dtype,              0},
    {"ndim",               1, nif_ndim,               0},
    {"size",               1, nif_size,               0},
    {"nbytes",             1, nif_nbytes,             0},

    // Array data (dirty schedulers for eval)
    {"eval",               1, nif_eval,               ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"to_binary",          1, nif_to_binary,          ERL_NIF_DIRTY_JOB_CPU_BOUND},

    // Unary ops
    {"mlx_negative",       2, nif_negative,           0},
    {"mlx_abs",            2, nif_abs,                0},
    {"mlx_sign",           2, nif_sign,               0},
    {"mlx_ceil",           2, nif_ceil,               0},
    {"mlx_floor",          2, nif_floor,              0},
    {"mlx_round",          2, nif_round,              0},
    {"mlx_exp",            2, nif_exp,                0},
    {"mlx_expm1",          2, nif_expm1,              0},
    {"mlx_log",            2, nif_log,                0},
    {"mlx_log2",           2, nif_log2,               0},
    {"mlx_log10",          2, nif_log10,              0},
    {"mlx_log1p",          2, nif_log1p,              0},
    {"mlx_sqrt",           2, nif_sqrt,               0},
    {"mlx_reciprocal",     2, nif_reciprocal,         0},
    {"mlx_sin",            2, nif_sin,                0},
    {"mlx_cos",            2, nif_cos,                0},
    {"mlx_tan",            2, nif_tan,                0},
    {"mlx_arcsin",         2, nif_arcsin,             0},
    {"mlx_arccos",         2, nif_arccos,             0},
    {"mlx_arctan",         2, nif_arctan,             0},
    {"mlx_sinh",           2, nif_sinh,               0},
    {"mlx_cosh",           2, nif_cosh,               0},
    {"mlx_tanh",           2, nif_tanh,               0},
    {"mlx_arcsinh",        2, nif_arcsinh,            0},
    {"mlx_arccosh",        2, nif_arccosh,            0},
    {"mlx_arctanh",        2, nif_arctanh,            0},
    {"mlx_erf",            2, nif_erf,                0},
    {"mlx_erfinv",         2, nif_erfinv,             0},
    {"mlx_logical_not",    2, nif_logical_not,        0},
    {"mlx_isnan",          2, nif_isnan,              0},
    {"mlx_isinf",          2, nif_isinf,              0},
    {"mlx_sigmoid",        2, nif_sigmoid,            0},
    {"mlx_bitwise_invert", 2, nif_bitwise_invert,    0},
    {"mlx_conjugate",      2, nif_conjugate,          0},
    {"mlx_real",           2, nif_real_op,            0},
    {"mlx_imag",           2, nif_imag_op,            0},

    // Wave 1: New unary ops
    {"mlx_rsqrt",          2, nif_rsqrt,              0},
    {"mlx_square",         2, nif_square,             0},
    {"mlx_degrees",        2, nif_degrees,            0},
    {"mlx_radians",        2, nif_radians,            0},
    {"mlx_isfinite",       2, nif_isfinite,           0},
    {"mlx_isneginf",       2, nif_isneginf,           0},
    {"mlx_isposinf",       2, nif_isposinf,           0},
    {"mlx_copy",           2, nif_copy,               0},
    {"mlx_stop_gradient",  2, nif_stop_gradient,      0},
    {"mlx_ones_like",      2, nif_ones_like,          0},
    {"mlx_zeros_like",     2, nif_zeros_like,         0},

    // Binary ops
    {"mlx_add",            3, nif_add,                0},
    {"mlx_subtract",       3, nif_subtract,           0},
    {"mlx_multiply",       3, nif_multiply,           0},
    {"mlx_divide",         3, nif_divide,             0},
    {"mlx_floor_divide",   3, nif_floor_divide,       0},
    {"mlx_power",          3, nif_power,              0},
    {"mlx_logaddexp",      3, nif_logaddexp,          0},
    {"mlx_arctan2",        3, nif_arctan2,            0},
    {"mlx_equal",          3, nif_equal,              0},
    {"mlx_not_equal",      3, nif_not_equal,          0},
    {"mlx_less",           3, nif_less,               0},
    {"mlx_less_equal",     3, nif_less_equal,         0},
    {"mlx_greater",        3, nif_greater,            0},
    {"mlx_greater_equal",  3, nif_greater_equal,      0},
    {"mlx_logical_and",    3, nif_logical_and,        0},
    {"mlx_logical_or",     3, nif_logical_or,         0},
    {"mlx_matmul",         3, nif_matmul,             0},
    {"mlx_maximum",        3, nif_maximum,            0},
    {"mlx_minimum",        3, nif_minimum,            0},

    // Wave 1: New binary ops
    {"mlx_remainder",      3, nif_remainder,          0},
    {"mlx_outer",          3, nif_outer,              0},
    {"mlx_inner",          3, nif_inner,              0},
    {"mlx_kron",           3, nif_kron,               0},

    // Reduction ops
    {"mlx_sum",            4, nif_sum,                0},
    {"mlx_prod",           4, nif_prod,               0},
    {"mlx_mean",           4, nif_mean,               0},
    {"mlx_min",            4, nif_min,                0},
    {"mlx_max",            4, nif_max,                0},
    {"mlx_argmax",         4, nif_argmax,             0},
    {"mlx_argmin",         4, nif_argmin,             0},
    {"mlx_all",            4, nif_all_op,             0},
    {"mlx_any",            4, nif_any_op,             0},

    // Wave 1: New reduction/cumulative ops
    {"mlx_logsumexp",      4, nif_logsumexp,          0},
    {"mlx_std",            5, nif_std,                0},
    {"mlx_var",            5, nif_var,                0},
    {"mlx_cumsum",         5, nif_cumsum,             0},
    {"mlx_cumprod",        5, nif_cumprod,            0},
    {"mlx_cummax",         5, nif_cummax,             0},
    {"mlx_cummin",         5, nif_cummin,             0},

    // Shape ops
    {"mlx_reshape",        3, nif_reshape,            0},
    {"mlx_transpose",      3, nif_transpose,          0},
    {"mlx_squeeze",        3, nif_squeeze,            0},
    {"mlx_expand_dims",    3, nif_expand_dims,        0},
    {"mlx_broadcast_to",   3, nif_broadcast_to,       0},
    {"mlx_flatten",        4, nif_flatten,            0},
    {"mlx_slice",          5, nif_slice,              0},

    // Type ops
    {"mlx_as_type",        3, nif_as_type,            0},

    // Selection ops
    {"mlx_where",          4, nif_where,              0},

    // Additional binary ops (bitwise, shifts)
    {"mlx_bitwise_and",    3, nif_bitwise_and,        0},
    {"mlx_bitwise_or",     3, nif_bitwise_or,         0},
    {"mlx_bitwise_xor",    3, nif_bitwise_xor,        0},
    {"mlx_left_shift",     3, nif_left_shift,         0},
    {"mlx_right_shift",    3, nif_right_shift,        0},

    // Clip, Sort, Take, Triangular, Pad, Repeat, Tile, Concatenate
    {"mlx_clip",           4, nif_clip,               0},
    {"mlx_sort",           3, nif_sort,               0},
    {"mlx_argsort",        3, nif_argsort,            0},
    {"mlx_take",           4, nif_take,               0},
    {"mlx_take_along_axis",4, nif_take_along_axis,    0},
    {"mlx_triu",           3, nif_triu,               0},
    {"mlx_tril",           3, nif_tril,               0},
    {"mlx_pad",            6, nif_pad,                0},
    {"mlx_repeat",         4, nif_repeat,             0},
    {"mlx_tile",           3, nif_tile,               0},
    {"mlx_concatenate",    3, nif_concatenate,         0},
    {"mlx_stack",          3, nif_stack,               0},
    {"mlx_slice_update",   6, nif_slice_update,        0},
    {"mlx_view",           3, nif_view,                0},

    // FFT ops
    {"mlx_fft",            4, nif_fft,                 0},
    {"mlx_ifft",           4, nif_ifft,                0},

    // Gather / Scatter ops
    {"mlx_gather",         5, nif_gather,              0},
    {"mlx_scatter",        5, nif_scatter,             0},
    {"mlx_scatter_add",    5, nif_scatter_add,         0},

    // Wave 1: New scatter variants
    {"mlx_scatter_max",    5, nif_scatter_max,         0},
    {"mlx_scatter_min",    5, nif_scatter_min,         0},
    {"mlx_scatter_prod",   5, nif_scatter_prod,        0},

    // Wave 1: Shape/Selection/Comparison/Matrix ops
    {"mlx_moveaxis",       4, nif_moveaxis,            0},
    {"mlx_swapaxes",       4, nif_swapaxes,            0},
    {"mlx_diagonal",       5, nif_diagonal,            0},
    {"mlx_diag",           3, nif_diag,                0},
    {"mlx_roll",           4, nif_roll,                0},
    {"mlx_unflatten",      4, nif_unflatten,           0},
    {"mlx_as_strided",     5, nif_as_strided,          0},
    {"mlx_identity",       3, nif_identity,            0},
    {"mlx_tri",            5, nif_tri,                 0},
    {"mlx_topk",           4, nif_topk,                0},
    {"mlx_partition",      4, nif_partition,           0},
    {"mlx_argpartition",   4, nif_argpartition,        0},
    {"mlx_put_along_axis", 5, nif_put_along_axis,      0},
    {"mlx_allclose",       5, nif_allclose,            0},
    {"mlx_isclose",        5, nif_isclose,             0},
    {"mlx_array_equal",    3, nif_array_equal,         0},
    {"mlx_nan_to_num",     5, nif_nan_to_num,          0},
    {"mlx_number_of_elements", 5, nif_number_of_elements, 0},
    {"mlx_softmax",        3, nif_softmax,             0},
    {"mlx_tensordot",      5, nif_tensordot,           0},
    {"mlx_addmm",          6, nif_addmm,               0},
    {"mlx_trace",          6, nif_trace,               0},
    {"mlx_einsum",         3, nif_einsum,              0},
    {"mlx_block_masked_mm", 7, nif_block_masked_mm,    0},

    // Convolution
    {"mlx_conv_general",  10, nif_conv_general,        0},

    // Split (for to_batched)
    {"mlx_split_equal_parts", 4, nif_split_equal_parts, 0},

    // Device / Stream
    {"default_cpu_stream", 0, nif_default_cpu_stream, 0},
    {"default_gpu_stream", 0, nif_default_gpu_stream, 0},
    {"default_device",     0, nif_default_device,     0},
    {"set_default_device", 1, nif_set_default_device, 0},
    {"device_new",         1, nif_device_new,         0},
    {"synchronize",        1, nif_synchronize,        ERL_NIF_DIRTY_JOB_CPU_BOUND},

    // Wave 2: Random ops
    {"mlx_random_key",              1, nif_random_key,              0},
    {"mlx_random_seed",             1, nif_random_seed,             0},
    {"mlx_random_split",            2, nif_random_split,            0},
    {"mlx_random_split_num",        3, nif_random_split_num,        0},
    {"mlx_random_uniform",          6, nif_random_uniform,          0},
    {"mlx_random_normal",           6, nif_random_normal,           0},
    {"mlx_random_bernoulli",        4, nif_random_bernoulli,        0},
    {"mlx_random_randint",          6, nif_random_randint,          0},
    {"mlx_random_truncated_normal", 6, nif_random_truncated_normal, 0},
    {"mlx_random_categorical",      4, nif_random_categorical,      0},
    {"mlx_random_gumbel",           4, nif_random_gumbel,           0},
    {"mlx_random_laplace",          6, nif_random_laplace,          0},

    // Wave 3: Linear Algebra ops
    {"mlx_linalg_inv",              2, nif_linalg_inv,              0},
    {"mlx_linalg_pinv",             2, nif_linalg_pinv,             0},
    {"mlx_linalg_cholesky",         3, nif_linalg_cholesky,         0},
    {"mlx_linalg_cholesky_inv",     3, nif_linalg_cholesky_inv,     0},
    {"mlx_linalg_qr",               2, nif_linalg_qr,               0},
    {"mlx_linalg_svd",              2, nif_linalg_svd,              0},
    {"mlx_linalg_eigh",             3, nif_linalg_eigh,             0},
    {"mlx_linalg_eigvalsh",         3, nif_linalg_eigvalsh,         0},
    {"mlx_linalg_lu",               2, nif_linalg_lu,               0},
    {"mlx_linalg_lu_factor",        2, nif_linalg_lu_factor,        0},
    {"mlx_linalg_solve",            3, nif_linalg_solve,            0},
    {"mlx_linalg_solve_triangular", 4, nif_linalg_solve_triangular, 0},
    {"mlx_linalg_cross",            4, nif_linalg_cross,            0},
    {"mlx_linalg_tri_inv",          3, nif_linalg_tri_inv,          0},
    {"mlx_linalg_norm",             4, nif_linalg_norm,             0},
    {"mlx_linalg_norm_p",           5, nif_linalg_norm_p,           0},

    // Wave 4: FFT expansion
    {"mlx_rfft",           4, nif_rfft,                0},
    {"mlx_irfft",          4, nif_irfft,               0},
    {"mlx_fft2",           4, nif_fft2,                0},
    {"mlx_fftn",           4, nif_fftn,                0},
    {"mlx_ifft2",          4, nif_ifft2,               0},
    {"mlx_ifftn",          4, nif_ifftn,               0},
    {"mlx_rfft2",          4, nif_rfft2,               0},
    {"mlx_rfftn",          4, nif_rfftn,               0},
    {"mlx_irfft2",         4, nif_irfft2,              0},
    {"mlx_irfftn",         4, nif_irfftn,              0},

    // Wave 5: Quantization
    {"mlx_quantize",              4, nif_quantize,             0},
    {"mlx_dequantize",            6, nif_dequantize,           0},
    {"mlx_quantized_matmul",      8, nif_quantized_matmul,     0},

    // Wave 6: Fast ops
    {"mlx_fast_layer_norm",       5, nif_fast_layer_norm,      0},
    {"mlx_fast_rms_norm",         4, nif_fast_rms_norm,        0},
    {"mlx_fast_rope",             8, nif_fast_rope,            0},
    {"mlx_fast_sdpa",             6, nif_fast_sdpa,            0},

    // Wave 7: I/O
    {"mlx_save",                  2, nif_save,                 0},
    {"mlx_load",                  2, nif_load,                 0},
    {"mlx_save_safetensors",      5, nif_save_safetensors,     0},
    {"mlx_load_safetensors",      2, nif_load_safetensors,     0},

    // Wave 9: Function transforms (closure bridge)
    {"closure_respond",           2, nif_closure_respond,      0},
    {"closure_respond_error",     1, nif_closure_respond_error, 0},
    {"value_and_grad_apply",      3, nif_value_and_grad_apply, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"vjp_apply",                 3, nif_vjp_apply,            ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"jvp_apply",                 3, nif_jvp_apply,            ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"vmap_apply",                4, nif_vmap_apply,           ERL_NIF_DIRTY_JOB_CPU_BOUND},
};

ERL_NIF_INIT(Elixir.Mlx.NIF, nif_funcs, load, NULL, upgrade, NULL)
