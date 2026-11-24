/*
 * Copyright (C) by Argonne National Laboratory
 * SPDX-License-Identifier: Apache-2.0
 */

/* GPU fastbox, CUDA implementation: the device pool (VMM memory shared as
 * POSIX fds), and the collective kernels over the per-group rings.
 *
 * Carve layout (one per group member, in that member's pool):
 *
 *   STATE_OFF   local state, touched only by this rank's kernels:
 *               [0] send count, [1] receive count, [2] barrier epoch
 *   CREDIT_OFF  receives consumed by the next rank (written by the next rank)
 *   BAR_OFF     barrier flags, [k] written by the round-k sender
 *   FLAG_OFF    per-slot sequence flags (written by the previous rank)
 *   SLOT_OFF    NSLOTS payload slots (written by the previous rank)
 *
 * Ring protocol, message k on the link prev -> me:
 *   sender:   wait until k - credit < NSLOTS; write the payload into my
 *             slot[k % NSLOTS] in the next rank's carve; fence; flag = k + 1.
 *   receiver: wait for flag[k % NSLOTS] == k + 1; read the payload; credit = k + 1
 *             in the previous rank's carve.
 * Counts never reset, so a stale flag cannot look new. Each kernel is one
 * thread block; thread 0 does the waiting and the flag stores.
 */

#include "mplconfig.h"
#include "mpl_err.h"
#include "mpl_gpu_cuda.h"
#include "mpl_gpu_fbox.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>

#define SLOT_SIZE   (64 * 1024)
#define NSLOTS      8
#define STATE_OFF   0
#define CREDIT_OFF  256
#define BAR_OFF     512
#define FLAG_OFF    1024
#define SLOT_OFF    4096
#define CARVE_SIZE  (SLOT_OFF + NSLOTS * SLOT_SIZE)

#define NTHREADS    512

#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif
#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif

#define CU_ERR_CHECK(call) \
    do { \
        CUresult r_ = (call); \
        if (r_ != CUDA_SUCCESS) { \
            const char *s_ = NULL; \
            cuGetErrorString(r_, &s_); \
            fprintf(stderr, "fbox: %s failed: %s\n", #call, s_ ? s_ : "?"); \
            return MPL_ERR_GPU_INTERNAL; \
        } \
    } while (0)

#define CUDA_ERR_CHECK(call) \
    do { \
        cudaError_t e_ = (call); \
        if (e_ != cudaSuccess) { \
            fprintf(stderr, "fbox: %s failed: %s\n", #call, cudaGetErrorString(e_)); \
            return MPL_ERR_GPU_INTERNAL; \
        } \
    } while (0)

/* ---- host: pool ---- */

static int get_cu_device(int dev_id, CUdevice * dev)
{
    if (dev_id < 0) {
        CUDA_ERR_CHECK(cudaGetDevice(&dev_id));
    }
    /* make sure the primary context exists and is current */
    CUDA_ERR_CHECK(cudaSetDevice(dev_id));
    CUDA_ERR_CHECK(cudaFree(0));
    CU_ERR_CHECK(cuDeviceGet(dev, dev_id));
    return MPL_SUCCESS;
}

static void fill_prop(CUmemAllocationProp * prop, CUdevice dev)
{
    memset(prop, 0, sizeof(*prop));
    prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop->location.id = dev;
    prop->requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
}

static int map_handle(CUmemGenericAllocationHandle h, size_t size, CUdevice dev, void **base)
{
    CUdeviceptr ptr;
    CU_ERR_CHECK(cuMemAddressReserve(&ptr, size, 0, 0, 0));
    CU_ERR_CHECK(cuMemMap(ptr, size, 0, h, 0));
    CUmemAccessDesc ad;
    memset(&ad, 0, sizeof(ad));
    ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ad.location.id = dev;
    ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_ERR_CHECK(cuMemSetAccess(ptr, size, &ad, 1));
    *base = (void *) ptr;
    return MPL_SUCCESS;
}

extern "C" size_t MPL_gpu_fbox_carve_size(void)
{
    return CARVE_SIZE;
}

extern "C" int MPL_gpu_fbox_pool_create(size_t size, int dev_id, void **base,
                                        size_t * actual_size, int *fd)
{
    int ret;
    CUdevice dev;
    ret = get_cu_device(dev_id, &dev);
    if (ret)
        return ret;

    int vmm = 0;
    CU_ERR_CHECK(cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                      dev));
    if (!vmm)
        return MPL_ERR_GPU_INTERNAL;

    CUmemAllocationProp prop;
    fill_prop(&prop, dev);
    size_t gran = 0;
    CU_ERR_CHECK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size = (size + gran - 1) / gran * gran;

    CUmemGenericAllocationHandle h;
    CU_ERR_CHECK(cuMemCreate(&h, size, &prop, 0));
    int exported_fd = -1;
    CU_ERR_CHECK(cuMemExportToShareableHandle(&exported_fd, h,
                                              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    ret = map_handle(h, size, dev, base);
    if (ret)
        return ret;

    *actual_size = size;
    *fd = exported_fd;
    return MPL_SUCCESS;
}

extern "C" int MPL_gpu_fbox_pool_import(int peer_pid, int peer_fd, size_t size, int dev_id,
                                        void **base)
{
    int ret;
    CUdevice dev;
    ret = get_cu_device(dev_id, &dev);
    if (ret)
        return ret;

    int pidfd = (int) syscall(SYS_pidfd_open, peer_pid, 0);
    if (pidfd < 0) {
        perror("fbox: pidfd_open");
        return MPL_ERR_GPU_INTERNAL;
    }
    int fd = (int) syscall(SYS_pidfd_getfd, pidfd, peer_fd, 0);
    close(pidfd);
    if (fd < 0) {
        perror("fbox: pidfd_getfd");
        return MPL_ERR_GPU_INTERNAL;
    }

    CUmemGenericAllocationHandle h;
    CUresult r = cuMemImportFromShareableHandle(&h, (void *) (uintptr_t) fd,
                                                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    close(fd);
    CU_ERR_CHECK(r);
    return map_handle(h, size, dev, base);
}

extern "C" int MPL_gpu_fbox_get_busid(int dev_id, char *busid, int len)
{
    if (dev_id < 0) {
        CUDA_ERR_CHECK(cudaGetDevice(&dev_id));
    }
    CUDA_ERR_CHECK(cudaDeviceGetPCIBusId(busid, len, dev_id));
    return MPL_SUCCESS;
}

extern "C" int MPL_gpu_fbox_can_access_peer(const char *peer_busid)
{
    int my_dev, peer_dev, can = 0;
    if (cudaGetDevice(&my_dev) != cudaSuccess)
        return 0;
    if (cudaDeviceGetByPCIBusId(&peer_dev, peer_busid) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    if (peer_dev == my_dev)
        return 1;
    if (cudaDeviceCanAccessPeer(&can, my_dev, peer_dev) != cudaSuccess)
        return 0;
    return can ? 1 : 0;
}

extern "C" int MPL_gpu_fbox_carve_init(void *carve)
{
    CUDA_ERR_CHECK(cudaMemset(carve, 0, CARVE_SIZE));
    CUDA_ERR_CHECK(cudaDeviceSynchronize());
    return MPL_SUCCESS;
}

/* ---- device: ring primitives ---- */

struct ring {
    volatile uint64_t *state;   /* local */
    volatile uint64_t *my_credit;       /* local, written by next */
    volatile uint64_t *prev_credit;     /* in prev's carve */
    volatile uint64_t *my_flags;        /* local, written by prev */
    volatile uint64_t *next_flags;      /* in next's carve */
    char *my_slots;
    char *next_slots;
};

__device__ static inline void ring_setup(struct ring *r, const MPL_gpu_fbox_group_t * g)
{
    int next = (g->rank + 1) % g->size;
    int prev = (g->rank + g->size - 1) % g->size;
    char *me = (char *) g->carve[g->rank];
    r->state = (volatile uint64_t *) (me + STATE_OFF);
    r->my_credit = (volatile uint64_t *) (me + CREDIT_OFF);
    r->prev_credit = (volatile uint64_t *) ((char *) g->carve[prev] + CREDIT_OFF);
    r->my_flags = (volatile uint64_t *) (me + FLAG_OFF);
    r->next_flags = (volatile uint64_t *) ((char *) g->carve[next] + FLAG_OFF);
    r->my_slots = me + SLOT_OFF;
    r->next_slots = (char *) g->carve[next] + SLOT_OFF;
}

/* Wait for room in the next rank's ring; returns the slot to fill. */
__device__ static inline char *send_begin(const struct ring *r, uint64_t k)
{
    if (threadIdx.x == 0) {
        while (k - *r->my_credit >= NSLOTS);
    }
    __syncthreads();
    return r->next_slots + (k % NSLOTS) * SLOT_SIZE;
}

__device__ static inline void send_end(const struct ring *r, uint64_t k)
{
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        r->next_flags[k % NSLOTS] = k + 1;
    }
}

/* Wait for message k from the previous rank; returns its slot. */
__device__ static inline const char *recv_begin(const struct ring *r, uint64_t k)
{
    if (threadIdx.x == 0) {
        while (r->my_flags[k % NSLOTS] != k + 1);
        __threadfence();
    }
    __syncthreads();
    return r->my_slots + (k % NSLOTS) * SLOT_SIZE;
}

__device__ static inline void recv_end(const struct ring *r, uint64_t k)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *r->prev_credit = k + 1;
    }
}

/* ---- device: element ops ---- */

template <typename T> struct ld_bits;
template <> struct ld_bits<float> { __device__ static float ld(const float *p) { return __ldcg(p); } };
template <> struct ld_bits<double> { __device__ static double ld(const double *p) { return __ldcg(p); } };
template <> struct ld_bits<int32_t> { __device__ static int32_t ld(const int32_t *p) { return __ldcg((const int *) p); } };
template <> struct ld_bits<int64_t> { __device__ static int64_t ld(const int64_t *p) { return __ldcg((const long long *) p); } };
template <> struct ld_bits<uint8_t> { __device__ static uint8_t ld(const uint8_t *p) { return __ldcg((const unsigned char *) p); } };
template <> struct ld_bits<__half> {
    __device__ static __half ld(const __half *p) { return __ushort_as_half(__ldcg((const unsigned short *) p)); }
};
template <> struct ld_bits<__nv_bfloat16> {
    __device__ static __nv_bfloat16 ld(const __nv_bfloat16 *p) {
        return __ushort_as_bfloat16(__ldcg((const unsigned short *) p));
    }
};

template <typename T> struct to_acc { typedef T type; };
template <> struct to_acc<__half> { typedef float type; };
template <> struct to_acc<__nv_bfloat16> { typedef float type; };

template <typename T> __device__ static inline typename to_acc<T>::type acc(T v) { return v; }
template <> __device__ inline float acc<__half>(__half v) { return __half2float(v); }
template <> __device__ inline float acc<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ static inline T from_acc(typename to_acc<T>::type v) { return (T) v; }
template <> __device__ inline __half from_acc<__half>(float v) { return __float2half(v); }
template <> __device__ inline __nv_bfloat16 from_acc<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

struct op_sum { template <typename A> __device__ static A f(A a, A b) { return a + b; } };
struct op_prod { template <typename A> __device__ static A f(A a, A b) { return a * b; } };
struct op_min { template <typename A> __device__ static A f(A a, A b) { return b < a ? b : a; } };
struct op_max { template <typename A> __device__ static A f(A a, A b) { return b > a ? b : a; } };
struct op_none { template <typename A> __device__ static A f(A a, A b) { return a; } };

template <typename T, typename OP>
__device__ static inline T combine(T a, T b)
{
    return from_acc<T>(OP::template f<typename to_acc<T>::type>(acc(a), acc(b)));
}

/* out1[i] (and out2[i] if not NULL) = slot ? OP(slot[i], user[i]) : user[i], where slot
 * is a ring slot (read through L2) and user may be NULL (then just slot[i]). */
template <typename T, typename OP>
__device__ static void move(T *out1, T *out2, const T *slot, const T *user, size_t n)
{
    const int VN = 16 / sizeof(T);
    bool vec = ((((uintptr_t) out1) | ((uintptr_t) out2) | ((uintptr_t) slot) |
                 ((uintptr_t) user)) % 16) == 0;
    size_t nvec = vec ? n / VN : 0;
    for (size_t i = threadIdx.x; i < nvec; i += blockDim.x) {
        uint4 a, b;
        if (slot)
            a = __ldcg((const uint4 *) slot + i);
        if (user)
            b = ((const uint4 *) user)[i];
        if (slot && user) {
            T *ea = (T *) & a;
            const T *eb = (const T *) &b;
#pragma unroll
            for (int j = 0; j < VN; j++)
                ea[j] = combine<T, OP>(ea[j], eb[j]);
        } else if (user) {
            a = b;
        }
        ((uint4 *) out1)[i] = a;
        if (out2)
            ((uint4 *) out2)[i] = a;
    }
    for (size_t i = nvec * VN + threadIdx.x; i < n; i += blockDim.x) {
        T v;
        if (slot && user)
            v = combine<T, OP>(ld_bits<T>::ld(slot + i), user[i]);
        else if (slot)
            v = ld_bits<T>::ld(slot + i);
        else
            v = user[i];
        out1[i] = v;
        if (out2)
            out2[i] = v;
    }
}

/* ---- device: collectives ---- */

/* Ring allreduce, one segment of size*E elements at a time (E elements per
 * slot). Message order on every link: own chunk, n-2 reduce-and-forward,
 * the completed chunk, n-2 allgather forwards; then n-1 receives of each kind. */
template <typename T, typename OP>
__global__ void fbox_allreduce_kernel(const T *sendbuf, T *recvbuf, size_t count,
                                      MPL_gpu_fbox_group_t g)
{
    __shared__ struct ring r;
    __shared__ uint64_t ks, kr;
    if (threadIdx.x == 0) {
        ring_setup(&r, &g);
        ks = r.state[0];
        kr = r.state[1];
    }
    __syncthreads();

    const int n = g.size, me = g.rank;
    const size_t E = SLOT_SIZE / sizeof(T);
    uint64_t s = ks, k = kr;

    for (size_t base = 0; base < count; base += (size_t) n * E) {
#define CHUNK(c, off, len) \
        size_t off = base + (size_t) (c) * E; \
        size_t len = off < count ? (count - off < E ? count - off : E) : 0;
        /* step 0: own chunk */
        {
            CHUNK(me, off, len);
            T *dst = (T *) send_begin(&r, s);
            move<T, OP>(dst, NULL, NULL, sendbuf + off, len);
            send_end(&r, s++);
        }
        /* reduce-scatter */
        for (int j = 1; j < n; j++) {
            int c = (me - j + n) % n;
            CHUNK(c, off, len);
            const T *src = (const T *) recv_begin(&r, k);
            T *dst = (T *) send_begin(&r, s);
            /* the last step completes chunk c: keep it and start the allgather */
            move<T, OP>(dst, j == n - 1 ? recvbuf + off : NULL, src, sendbuf + off, len);
            recv_end(&r, k++);
            send_end(&r, s++);
        }
        /* allgather */
        for (int j = 1; j < n; j++) {
            int c = (me + 1 - j + n) % n;
            CHUNK(c, off, len);
            const T *src = (const T *) recv_begin(&r, k);
            if (j < n - 1) {
                T *dst = (T *) send_begin(&r, s);
                move<T, op_none>(dst, recvbuf + off, src, NULL, len);
                send_end(&r, s++);
            } else {
                move<T, op_none>(recvbuf + off, NULL, src, NULL, len);
            }
            recv_end(&r, k++);
        }
#undef CHUNK
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        r.state[0] = s;
        r.state[1] = k;
    }
}

/* Chain broadcast from the root along the ring, one slot at a time. */
__global__ void fbox_bcast_kernel(uint8_t *buf, size_t nbytes, int root, MPL_gpu_fbox_group_t g)
{
    __shared__ struct ring r;
    __shared__ uint64_t ks, kr;
    if (threadIdx.x == 0) {
        ring_setup(&r, &g);
        ks = r.state[0];
        kr = r.state[1];
    }
    __syncthreads();

    const int n = g.size;
    const int pos = (g.rank - root + n) % n;
    uint64_t s = ks, k = kr;

    for (size_t off = 0; off < nbytes; off += SLOT_SIZE) {
        size_t len = nbytes - off < SLOT_SIZE ? nbytes - off : SLOT_SIZE;
        if (pos == 0) {
            uint8_t *dst = (uint8_t *) send_begin(&r, s);
            move<uint8_t, op_none>(dst, NULL, NULL, buf + off, len);
            send_end(&r, s++);
        } else {
            const uint8_t *src = (const uint8_t *) recv_begin(&r, k);
            if (pos < n - 1) {
                uint8_t *dst = (uint8_t *) send_begin(&r, s);
                move<uint8_t, op_none>(dst, buf + off, src, NULL, len);
                send_end(&r, s++);
            } else {
                move<uint8_t, op_none>(buf + off, NULL, src, NULL, len);
            }
            recv_end(&r, k++);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        r.state[0] = s;
        r.state[1] = k;
    }
}

/* Dissemination barrier: round i stores the epoch into flag i of rank + 2^i and
 * waits for flag i of its own carve. One thread. */
__global__ void fbox_barrier_kernel(MPL_gpu_fbox_group_t g)
{
    volatile uint64_t *state = (volatile uint64_t *) ((char *) g.carve[g.rank] + STATE_OFF);
    volatile uint64_t *my_bar = (volatile uint64_t *) ((char *) g.carve[g.rank] + BAR_OFF);
    uint64_t e = state[2] + 1;
    state[2] = e;
    for (int i = 0, d = 1; d < g.size; i++, d <<= 1) {
        int peer = (g.rank + d) % g.size;
        volatile uint64_t *peer_bar =
            (volatile uint64_t *) ((char *) g.carve[peer] + BAR_OFF);
        __threadfence_system();
        peer_bar[i] = e;
        while (my_bar[i] < e);
    }
    __threadfence_system();
}

/* ---- host: launch ---- */

template <typename T, typename OP>
static int launch_allreduce(const void *sendbuf, void *recvbuf, size_t count,
                            const MPL_gpu_fbox_group_t * g, cudaStream_t stream)
{
    fbox_allreduce_kernel<T, OP> <<<1, NTHREADS, 0, stream>>> ((const T *) sendbuf, (T *) recvbuf,
                                                               count, *g);
    CUDA_ERR_CHECK(cudaGetLastError());
    return MPL_SUCCESS;
}

template <typename T>
static int launch_allreduce_op(const void *sendbuf, void *recvbuf, size_t count,
                               MPL_gpu_fbox_op_t op, const MPL_gpu_fbox_group_t * g,
                               cudaStream_t stream)
{
    switch (op) {
        case MPL_GPU_FBOX_SUM:
            return launch_allreduce<T, op_sum>(sendbuf, recvbuf, count, g, stream);
        case MPL_GPU_FBOX_PROD:
            return launch_allreduce<T, op_prod>(sendbuf, recvbuf, count, g, stream);
        case MPL_GPU_FBOX_MIN:
            return launch_allreduce<T, op_min>(sendbuf, recvbuf, count, g, stream);
        case MPL_GPU_FBOX_MAX:
            return launch_allreduce<T, op_max>(sendbuf, recvbuf, count, g, stream);
        default:
            return MPL_ERR_GPU_INTERNAL;
    }
}

extern "C" int MPL_gpu_fbox_allreduce(const void *sendbuf, void *recvbuf, size_t count,
                                      MPL_gpu_fbox_type_t type, MPL_gpu_fbox_op_t op,
                                      const MPL_gpu_fbox_group_t * g, MPL_gpu_stream_t stream)
{
    static const size_t type_size[] = { 4, 8, 2, 2, 4, 8, 1 };
    if (count == 0)
        return MPL_SUCCESS;
    if (g->size == 1) {
        if (sendbuf != recvbuf) {
            CUDA_ERR_CHECK(cudaMemcpyAsync(recvbuf, sendbuf, count * type_size[type],
                                           cudaMemcpyDeviceToDevice, stream));
        }
        return MPL_SUCCESS;
    }
    switch (type) {
        case MPL_GPU_FBOX_FLOAT:
            return launch_allreduce_op<float>(sendbuf, recvbuf, count, op, g, stream);
        case MPL_GPU_FBOX_DOUBLE:
            return launch_allreduce_op<double>(sendbuf, recvbuf, count, op, g, stream);
        case MPL_GPU_FBOX_HALF:
            return launch_allreduce_op<__half>(sendbuf, recvbuf, count, op, g, stream);
        case MPL_GPU_FBOX_BF16:
            return launch_allreduce_op<__nv_bfloat16>(sendbuf, recvbuf, count, op, g, stream);
        case MPL_GPU_FBOX_INT32:
            return launch_allreduce_op<int32_t>(sendbuf, recvbuf, count, op, g, stream);
        case MPL_GPU_FBOX_INT64:
            return launch_allreduce_op<int64_t>(sendbuf, recvbuf, count, op, g, stream);
        case MPL_GPU_FBOX_UINT8:
            return launch_allreduce_op<uint8_t>(sendbuf, recvbuf, count, op, g, stream);
        default:
            return MPL_ERR_GPU_INTERNAL;
    }
}

extern "C" int MPL_gpu_fbox_bcast(void *buf, size_t nbytes, int root,
                                  const MPL_gpu_fbox_group_t * g, MPL_gpu_stream_t stream)
{
    if (nbytes == 0 || g->size == 1)
        return MPL_SUCCESS;
    fbox_bcast_kernel <<<1, NTHREADS, 0, stream>>> ((uint8_t *) buf, nbytes, root, *g);
    CUDA_ERR_CHECK(cudaGetLastError());
    return MPL_SUCCESS;
}

extern "C" int MPL_gpu_fbox_barrier(const MPL_gpu_fbox_group_t * g, MPL_gpu_stream_t stream)
{
    if (g->size == 1)
        return MPL_SUCCESS;
    fbox_barrier_kernel <<<1, 1, 0, stream>>> (*g);
    CUDA_ERR_CHECK(cudaGetLastError());
    return MPL_SUCCESS;
}
