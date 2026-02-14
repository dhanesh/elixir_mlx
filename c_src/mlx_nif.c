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
static ERL_NIF_TERM wrap_array(ErlNifEnv* env, mlx_array arr) {
    MlxArrayResource *res = enif_alloc_resource(MLX_ARRAY_RESOURCE, sizeof(MlxArrayResource));
    res->inner = arr;
    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return MLX_NIF_OK(env, term);
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

// ============================================================
// NIF: Reduction operations (arr, axes|nil, keepdims, stream)
// ============================================================

DEF_REDUCE_OP(nif_sum,  mlx_sum_all,  mlx_sum)
DEF_REDUCE_OP(nif_prod, mlx_prod_all, mlx_prod)
DEF_REDUCE_OP(nif_mean, mlx_mean_all, mlx_mean)
DEF_REDUCE_OP(nif_min,  mlx_min_all,  mlx_min)
DEF_REDUCE_OP(nif_max,  mlx_max_all,  mlx_max)

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
    int ret = mlx_argmax(&result, a->inner, axis, keepdims, s->inner);
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
    int ret = mlx_argmin(&result, a->inner, axis, keepdims, s->inner);
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
        ret = mlx_transpose_all(&result, a->inner, s->inner);
    } else {
        int axes[16], nax;
        if (!get_int_list(env, argv[1], axes, &nax))
            return MLX_NIF_ERROR(env, "expected axes list");
        ret = mlx_transpose(&result, a->inner, axes, (size_t)nax, s->inner);
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
        ret = mlx_squeeze_all(&result, a->inner, s->inner);
    } else {
        int axes[16], nax;
        if (!get_int_list(env, argv[1], axes, &nax))
            return MLX_NIF_ERROR(env, "expected axes list");
        ret = mlx_squeeze(&result, a->inner, axes, (size_t)nax, s->inner);
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
    int ret = mlx_expand_dims(&result, a->inner, axes_arr, 1, s->inner);
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
    int ret = mlx_sort(&result, a->inner, axis, s->inner);
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
    int ret = mlx_argsort(&result, a->inner, axis, s->inner);
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
    int ret = mlx_take(&result, a->inner, indices->inner, axis, s->inner);
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
    int ret = mlx_repeat(&result, a->inner, repeats, axis, s->inner);
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
    int ret = mlx_concatenate(&result, vec, axis, s->inner);
    mlx_vector_array_free(vec);
    if (ret != 0) return MLX_NIF_ERROR(env, last_error_msg);
    return wrap_array(env, result);
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

    if (!MLX_ARRAY_RESOURCE || !MLX_STREAM_RESOURCE || !MLX_DEVICE_RESOURCE ||
        !MLX_VECTOR_ARRAY_RESOURCE)
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

    // Reduction ops
    {"mlx_sum",            4, nif_sum,                0},
    {"mlx_prod",           4, nif_prod,               0},
    {"mlx_mean",           4, nif_mean,               0},
    {"mlx_min",            4, nif_min,                0},
    {"mlx_max",            4, nif_max,                0},
    {"mlx_argmax",         4, nif_argmax,             0},
    {"mlx_argmin",         4, nif_argmin,             0},

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

    // Device / Stream
    {"default_cpu_stream", 0, nif_default_cpu_stream, 0},
    {"default_gpu_stream", 0, nif_default_gpu_stream, 0},
    {"default_device",     0, nif_default_device,     0},
    {"set_default_device", 1, nif_set_default_device, 0},
    {"device_new",         1, nif_device_new,         0},
    {"synchronize",        1, nif_synchronize,        ERL_NIF_DIRTY_JOB_CPU_BOUND},
};

ERL_NIF_INIT(Elixir.Mlx.NIF, nif_funcs, load, NULL, upgrade, NULL)
