/*
 * Copyright (C) by Argonne National Laboratory
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mpiimpl.h"

/* GPU fastbox glue (see mpl_gpu_fbox.h for the transport).
 *
 * At MPIX_Stream_comm_create on a GPU stream, if every rank of the comm is on
 * this node, drives one GPU, and can reach every other rank's GPU peer to
 * peer, the new comm gets an fbox group and its enqueued allreduce and bcast
 * run as device kernels. Otherwise the comm uses the host-staged path.
 *
 * Setup is lazy and collective over the comm being created: a process creates
 * its device pool on first use (on the current device), and maps a peer's pool
 * the first time a comm includes that peer. Pool fds are passed with
 * pidfd_getfd. A comm's carve is retired at free, not reused, because no rank
 * can tell when its peers have stopped writing into it.
 *
 * Environment (temporary, until CVARs):
 *   MPIR_CVAR_GPU_FBOX=0              turn the transport off
 *   MPIR_CVAR_GPU_FBOX_SHARED_DEVICE=1  allow ranks that share one GPU (testing)
 *   MPIR_CVAR_GPU_FBOX_VERBOSE=1      report the decision at comm create
 */

#ifdef MPL_HAVE_CUDA

#define FBOX_NUM_CARVES 32
#define FBOX_BUSID_LEN 32

static struct {
    bool tried;                 /* pool creation attempted */
    bool ok;                    /* pool available */
    void *base;
    size_t size;
    int fd;
    int dev;
    char busid[FBOX_BUSID_LEN];
    bool carve_used[FBOX_NUM_CARVES];
    void **peer_base;           /* indexed by world rank */
} fbox;

struct fbox_info {
    int ok;
    int node_id;
    int world_rank;
    int pid;
    int fd;
    int carve;
    size_t pool_size;
    char busid[FBOX_BUSID_LEN];
};

static int env_int(const char *name, int dflt)
{
    int val = dflt;
    MPL_env2int(name, &val);
    return val;
}

static bool pool_init(void)
{
    if (fbox.tried) {
        return fbox.ok;
    }
    fbox.tried = true;

    size_t want = MPL_gpu_fbox_carve_size() * FBOX_NUM_CARVES;
    if (MPL_gpu_fbox_pool_create(want, -1, &fbox.base, &fbox.size, &fbox.fd) != MPL_SUCCESS) {
        return false;
    }
    if (MPL_gpu_fbox_get_busid(-1, fbox.busid, FBOX_BUSID_LEN) != MPL_SUCCESS) {
        return false;
    }
    cudaGetDevice(&fbox.dev);
    fbox.peer_base = MPL_calloc(MPIR_Process.size, sizeof(void *), MPL_MEM_OTHER);
    if (!fbox.peer_base) {
        return false;
    }
    fbox.peer_base[MPIR_Process.rank] = fbox.base;
    fbox.ok = true;
    return true;
}

static int carve_get(void)
{
    for (int i = 0; i < FBOX_NUM_CARVES; i++) {
        if (!fbox.carve_used[i]) {
            fbox.carve_used[i] = true;
            return i;
        }
    }
    return -1;
}

static void *carve_addr(void *base, int carve)
{
    return (char *) base + (size_t) carve *MPL_gpu_fbox_carve_size();
}

/* All checks that every rank can do on the gathered info, so all reach the same answer. */
static bool group_eligible(struct fbox_info *info, int n, bool shared_ok)
{
    if (n > MPL_GPU_FBOX_MAX_RANKS) {
        return false;
    }
    for (int i = 0; i < n; i++) {
        if (!info[i].ok || info[i].node_id != info[0].node_id) {
            return false;
        }
        if (!shared_ok) {
            for (int j = 0; j < i; j++) {
                if (strcmp(info[i].busid, info[j].busid) == 0) {
                    return false;
                }
            }
        }
    }
    return true;
}

int MPIR_Stream_comm_fbox_init(MPIR_Comm * parent, MPIR_Comm * newcomm, MPIR_Stream * stream_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    int n = parent->local_size;
    struct fbox_info my, *info = NULL;
    MPL_gpu_fbox_group_t *group = NULL;
    int verbose = env_int("MPIR_CVAR_GPU_FBOX_VERBOSE", 0);
    const char *why = NULL;

    newcomm->stream_comm.single.fbox = NULL;

    /* Same decision on every rank: only CVAR-like settings that must be uniform. */
    if (!env_int("MPIR_CVAR_GPU_FBOX", 1) || !MPIR_CVAR_ENABLE_GPU ||
        parent->comm_kind != MPIR_COMM_KIND__INTRACOMM || n > MPL_GPU_FBOX_MAX_RANKS) {
        goto fn_exit;
    }

    memset(&my, 0, sizeof(my));
    my.carve = -1;
    if (stream_ptr && stream_ptr->type == MPIR_STREAM_GPU && pool_init()) {
        my.carve = carve_get();
    }
    my.ok = (my.carve >= 0);
    my.node_id = MPIR_Process.node_map ? MPIR_Process.node_map[MPIR_Process.rank] : 0;
    my.world_rank = MPIR_Process.rank;
    my.pid = getpid();
    my.fd = fbox.fd;
    my.pool_size = fbox.size;
    MPL_strncpy(my.busid, fbox.busid, FBOX_BUSID_LEN);
    if (my.ok && MPL_gpu_fbox_carve_init(carve_addr(fbox.base, my.carve)) != MPL_SUCCESS) {
        my.ok = 0;
    }

    info = MPL_malloc(n * sizeof(struct fbox_info), MPL_MEM_OTHER);
    MPIR_ERR_CHKANDJUMP(!info, mpi_errno, MPI_ERR_OTHER, "**nomem");
    mpi_errno = MPIR_Allgather_fallback(&my, sizeof(my), MPIR_BYTE_INTERNAL,
                                        info, sizeof(my), MPIR_BYTE_INTERNAL, parent,
                                        MPIR_COLL_ATTR_SYNC);
    MPIR_ERR_CHECK(mpi_errno);

    bool shared_ok = env_int("MPIR_CVAR_GPU_FBOX_SHARED_DEVICE", 0);
    if (!group_eligible(info, n, shared_ok)) {
        why = "not all ranks are on one node with a distinct GPU";
        goto release;
    }

    /* Peer access and mapping can fail locally; agree on the result below. */
    int local_ok = 1;
    for (int i = 0; i < n && local_ok; i++) {
        if (i == parent->rank) {
            continue;
        }
        if (!shared_ok && MPL_gpu_fbox_can_access_peer(info[i].busid) == 0) {
            local_ok = 0;
            break;
        }
        int wr = info[i].world_rank;
        if (!fbox.peer_base[wr]) {
            if (MPL_gpu_fbox_pool_import(info[i].pid, info[i].fd, info[i].pool_size, fbox.dev,
                                         &fbox.peer_base[wr]) != MPL_SUCCESS) {
                fbox.peer_base[wr] = NULL;
                local_ok = 0;
            }
        }
    }
    int all_ok = 0;
    mpi_errno = MPIR_Allreduce_fallback(&local_ok, &all_ok, 1, MPIR_INT_INTERNAL, MPI_MIN,
                                        parent, MPIR_COLL_ATTR_SYNC);
    MPIR_ERR_CHECK(mpi_errno);
    if (!all_ok) {
        why = "peer access or pool mapping failed";
        goto release;
    }

    group = MPL_malloc(sizeof(MPL_gpu_fbox_group_t), MPL_MEM_OTHER);
    MPIR_ERR_CHKANDJUMP(!group, mpi_errno, MPI_ERR_OTHER, "**nomem");
    group->rank = parent->rank;
    group->size = n;
    for (int i = 0; i < n; i++) {
        group->carve[i] = carve_addr(fbox.peer_base[info[i].world_rank], info[i].carve);
    }
    newcomm->stream_comm.single.fbox = group;
    newcomm->stream_comm.single.fbox_carve = my.carve;
    if (verbose && parent->rank == 0) {
        printf("fbox: enabled on a %d-rank stream comm\n", n);
    }
    if (env_int("MPIR_CVAR_GPU_FBOX_SELFTEST", 0)) {
        /* time device barriers on the comm's stream */
        const int iters = 1000;
        MPL_gpu_stream_t s = stream_ptr->u.gpu_stream;
        MPL_gpu_fbox_barrier(group, s);
        cudaStreamSynchronize(s);
        MPL_time_t t0, t1;
        MPL_wtime(&t0);
        for (int i = 0; i < iters; i++) {
            MPL_gpu_fbox_barrier(group, s);
        }
        cudaError_t cerr = cudaStreamSynchronize(s);
        MPL_wtime(&t1);
        double dt;
        MPL_wtime_diff(&t0, &t1, &dt);
        if (parent->rank == 0) {
            printf("fbox: selftest: %d device barriers, %.2f us each (%s)\n", iters,
                   dt * 1e6 / iters, cerr == cudaSuccess ? "ok" : cudaGetErrorString(cerr));
        }
    }
    goto fn_exit;

  release:
    if (my.carve >= 0) {
        fbox.carve_used[my.carve] = false;
    }
    if (verbose && parent->rank == 0) {
        printf("fbox: disabled on a %d-rank stream comm: %s\n", n, why);
    }

  fn_exit:
    MPL_free(info);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

void MPIR_Stream_comm_fbox_free(MPIR_Comm * comm)
{
    if (comm->stream_comm.single.fbox) {
        /* Wait for this rank's kernels. The carve stays marked used (retired). */
        if (comm->stream_comm.single.stream) {
            cudaStreamSynchronize(comm->stream_comm.single.stream->u.gpu_stream);
        }
        MPL_free(comm->stream_comm.single.fbox);
        comm->stream_comm.single.fbox = NULL;
    }
}

static bool fbox_type(MPI_Datatype dt, MPL_gpu_fbox_type_t * t)
{
    switch (MPIR_DATATYPE_GET_RAW_INTERNAL(dt)) {
        case MPIR_FLOAT32:
            *t = MPL_GPU_FBOX_FLOAT;
            return true;
        case MPIR_FLOAT64:
            *t = MPL_GPU_FBOX_DOUBLE;
            return true;
        case MPIR_FLOAT16:
            *t = MPL_GPU_FBOX_HALF;
            return true;
        case MPIR_BFLOAT16:
            *t = MPL_GPU_FBOX_BF16;
            return true;
        case MPIR_INT32:
            *t = MPL_GPU_FBOX_INT32;
            return true;
        case MPIR_INT64:
            *t = MPL_GPU_FBOX_INT64;
            return true;
        case MPIR_UINT8:
            *t = MPL_GPU_FBOX_UINT8;
            return true;
        default:
            return false;
    }
}

static bool fbox_op(MPI_Op op, MPL_gpu_fbox_op_t * o)
{
    if (op == MPI_SUM) {
        *o = MPL_GPU_FBOX_SUM;
    } else if (op == MPI_PROD) {
        *o = MPL_GPU_FBOX_PROD;
    } else if (op == MPI_MIN) {
        *o = MPL_GPU_FBOX_MIN;
    } else if (op == MPI_MAX) {
        *o = MPL_GPU_FBOX_MAX;
    } else {
        return false;
    }
    return true;
}

static bool is_dev(const void *ptr)
{
    MPL_pointer_attr_t attr;
    MPIR_GPU_query_pointer_attr(ptr, &attr);
    return attr.type == MPL_GPU_POINTER_DEV;
}

/* Returns true if the operation was enqueued on the device. */
bool MPIR_Stream_fbox_allreduce(const void *sendbuf, void *recvbuf, MPI_Aint count,
                                MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm_ptr,
                                MPL_gpu_stream_t stream, int *mpi_errno)
{
    MPL_gpu_fbox_group_t *g = comm_ptr->stream_comm.single.fbox;
    MPL_gpu_fbox_type_t t;
    MPL_gpu_fbox_op_t o;

    if (comm_ptr->stream_comm_type != MPIR_STREAM_COMM_SINGLE || !g) {
        return false;
    }
    if (!fbox_type(datatype, &t) || !fbox_op(op, &o)) {
        return false;
    }
    if (sendbuf == MPI_IN_PLACE) {
        sendbuf = recvbuf;
    }
    if (count > 0 && (!is_dev(sendbuf) || !is_dev(recvbuf))) {
        return false;
    }
    if (MPL_gpu_fbox_allreduce(sendbuf, recvbuf, count, t, o, g, stream) != MPL_SUCCESS) {
        *mpi_errno = MPI_ERR_OTHER;
    }
    return true;
}

bool MPIR_Stream_fbox_bcast(void *buffer, MPI_Aint count, MPI_Datatype datatype, int root,
                            MPIR_Comm * comm_ptr, MPL_gpu_stream_t stream, int *mpi_errno)
{
    MPL_gpu_fbox_group_t *g = comm_ptr->stream_comm.single.fbox;

    if (comm_ptr->stream_comm_type != MPIR_STREAM_COMM_SINGLE || !g) {
        return false;
    }
    int is_contig;
    MPIR_Datatype_is_contig(datatype, &is_contig);
    if (!is_contig) {
        return false;
    }
    MPI_Aint dt_size, true_lb;
    MPIR_Datatype_get_size_macro(datatype, dt_size);
    MPIR_Datatype_get_true_lb(datatype, &true_lb);
    char *buf = (char *) buffer + true_lb;
    if (count > 0 && !is_dev(buf)) {
        return false;
    }
    if (MPL_gpu_fbox_bcast(buf, (size_t) count * dt_size, root, g, stream) != MPL_SUCCESS) {
        *mpi_errno = MPI_ERR_OTHER;
    }
    return true;
}

#else /* !MPL_HAVE_CUDA */

int MPIR_Stream_comm_fbox_init(MPIR_Comm * parent, MPIR_Comm * newcomm, MPIR_Stream * stream_ptr)
{
    newcomm->stream_comm.single.fbox = NULL;
    return MPI_SUCCESS;
}

void MPIR_Stream_comm_fbox_free(MPIR_Comm * comm)
{
}

bool MPIR_Stream_fbox_allreduce(const void *sendbuf, void *recvbuf, MPI_Aint count,
                                MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm_ptr,
                                MPL_gpu_stream_t stream, int *mpi_errno)
{
    return false;
}

bool MPIR_Stream_fbox_bcast(void *buffer, MPI_Aint count, MPI_Datatype datatype, int root,
                            MPIR_Comm * comm_ptr, MPL_gpu_stream_t stream, int *mpi_errno)
{
    return false;
}

#endif /* MPL_HAVE_CUDA */
