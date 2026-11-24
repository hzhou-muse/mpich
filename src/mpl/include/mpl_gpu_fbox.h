/*
 * Copyright (C) by Argonne National Laboratory
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MPL_GPU_FBOX_H_INCLUDED
#define MPL_GPU_FBOX_H_INCLUDED

/* GPU fastbox: an intranode transport in device memory that GPU kernels use
 * directly, so collectives run entirely on the device.
 *
 * Each process allocates one device pool (VMM memory, exportable as a POSIX
 * fd) and maps the pools of its peers once. A group of ranks (a stream
 * communicator in MPICH) gets one "carve" of every member's pool. A carve
 * holds the receive ring from the previous rank in the group (rank - 1), the
 * credit word written by the next rank, and the flags of a device barrier.
 * All remote traffic is stores; every rank polls only its own memory.
 *
 * The caller (MPICH) exchanges pids, fds and carve offsets, and fills an
 * MPL_gpu_fbox_group_t with the device address of every member's carve. */

#define MPL_GPU_FBOX_MAX_RANKS 64

typedef enum {
    MPL_GPU_FBOX_FLOAT,
    MPL_GPU_FBOX_DOUBLE,
    MPL_GPU_FBOX_HALF,
    MPL_GPU_FBOX_BF16,
    MPL_GPU_FBOX_INT32,
    MPL_GPU_FBOX_INT64,
    MPL_GPU_FBOX_UINT8,
    MPL_GPU_FBOX_NUM_TYPES
} MPL_gpu_fbox_type_t;

typedef enum {
    MPL_GPU_FBOX_SUM,
    MPL_GPU_FBOX_PROD,
    MPL_GPU_FBOX_MIN,
    MPL_GPU_FBOX_MAX,
    MPL_GPU_FBOX_NUM_OPS
} MPL_gpu_fbox_op_t;

typedef struct {
    int rank;
    int size;
    void *carve[MPL_GPU_FBOX_MAX_RANKS];        /* device address of each member's carve,
                                                 * mapped in this process */
} MPL_gpu_fbox_group_t;

#ifdef MPL_HAVE_CUDA

#ifdef __cplusplus
extern "C" {
#endif

/* Size of one carve (bytes). */
size_t MPL_gpu_fbox_carve_size(void);

/* Allocate a pool of at least `size` bytes on device `dev_id` (current device if
 * negative), map it, and export it. Returns the base address, the actual size and
 * an fd that stays open for the life of the process. */
int MPL_gpu_fbox_pool_create(size_t size, int dev_id, void **base, size_t * actual_size,
                             int *fd);

/* Map a peer's pool, given the peer's pid and the fd it exported. */
int MPL_gpu_fbox_pool_import(int peer_pid, int peer_fd, size_t size, int dev_id, void **base);

/* PCI bus id of device `dev_id` (current device if negative), and whether this
 * process's device can access the device with that bus id peer to peer.
 * Returns 1 if accessible, 0 if not, -1 if the device is not visible here. */
int MPL_gpu_fbox_get_busid(int dev_id, char *busid, int len);
int MPL_gpu_fbox_can_access_peer(const char *peer_busid);

/* Zero a carve (synchronously). */
int MPL_gpu_fbox_carve_init(void *carve);

/* Collectives, enqueued on `stream`. Buffers are device memory. Every member
 * must issue the same sequence of calls on its group. */
int MPL_gpu_fbox_allreduce(const void *sendbuf, void *recvbuf, size_t count,
                           MPL_gpu_fbox_type_t type, MPL_gpu_fbox_op_t op,
                           const MPL_gpu_fbox_group_t * group, MPL_gpu_stream_t stream);
int MPL_gpu_fbox_bcast(void *buf, size_t nbytes, int root, const MPL_gpu_fbox_group_t * group,
                       MPL_gpu_stream_t stream);
int MPL_gpu_fbox_barrier(const MPL_gpu_fbox_group_t * group, MPL_gpu_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* MPL_HAVE_CUDA */

#endif /* MPL_GPU_FBOX_H_INCLUDED */
