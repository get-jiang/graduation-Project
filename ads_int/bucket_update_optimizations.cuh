#ifndef ADS_INT_BUCKET_UPDATE_OPTIMIZATIONS_CUH_
#define ADS_INT_BUCKET_UPDATE_OPTIMIZATIONS_CUH_

#include "common.h"
#include "wl.h"

// Approximate ring order: smaller forward distance from current source bucket means earlier.
__device__ __forceinline__ unsigned ring_rank_from_src(unsigned src_bag_id, unsigned target_bag_id) {
    return (target_bag_id + NUM_BAG - src_bag_id) % NUM_BAG;
}

// same-bucket hysteresis: only re-activate when candidate bucket is strictly better than
// the currently registered bucket for this vertex (under current src bucket frame).
__device__ __forceinline__ bool should_reactivate_vertex(unsigned* registered_bucket, unsigned src_bag_id,
                                                         unsigned vertex, unsigned candidate_bag_id) {
    unsigned old_bag = atomicAdd(&(registered_bucket[vertex]), 0);
    unsigned cand_rank = ring_rank_from_src(src_bag_id, candidate_bag_id);
    unsigned old_rank = ring_rank_from_src(src_bag_id, old_bag);

    if (cand_rank >= old_rank) {
        return false;
    }

    unsigned observed = atomicCAS(&(registered_bucket[vertex]), old_bag, candidate_bag_id);
    while (observed != old_bag) {
        old_bag = observed;
        old_rank = ring_rank_from_src(src_bag_id, old_bag);
        if (cand_rank >= old_rank) {
            return false;
        }
        observed = atomicCAS(&(registered_bucket[vertex]), old_bag, candidate_bag_id);
    }
    return true;
}

// lazy bucket update + batched re-bucketing inside one warp:
// deduplicate same destination vertex and only let one lane do atomicMin/push.
__device__ __forceinline__ void relax_edge_batched(CSRGraph graph, worklist& wl, unsigned src_bag_id,
                                                   index_type dst, node_data_type new_dist,
                                                   unsigned* registered_bucket) {
    unsigned active = __activemask();
    unsigned group = __match_any_sync(active, (unsigned) dst);
    int leader = find_ms_bit(group);

    node_data_type best_dist = new_dist;
    unsigned lanes = group;
    while (lanes) {
        int lane = __ffs(lanes) - 1;
        node_data_type dist_lane = __shfl_sync(active, new_dist, lane);
        best_dist = min(best_dist, dist_lane);
        lanes &= (lanes - 1);
    }

    if ((int) get_lane_id() != leader) {
        return;
    }

    node_data_type old_dist = atomicMin(&(graph.node_data[dst]), best_dist);
    if (old_dist <= best_dist) {
        return;
    }

    unsigned dst_bag_id = wl.dist_to_bag_id_int(src_bag_id, best_dist);
    if (should_reactivate_vertex(registered_bucket, src_bag_id, dst, dst_bag_id)) {
        wl.push_work(dst_bag_id, dst);
    }
}

#endif
