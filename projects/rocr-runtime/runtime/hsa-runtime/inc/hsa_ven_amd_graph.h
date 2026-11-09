/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_INC_HSA_VEN_AMD_GRAPH_H_
#define HSA_RUNTIME_INC_HSA_VEN_AMD_GRAPH_H_

#include "hsa.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HSA_VEN_AMD_GRAPH_VERSION_MAJOR 0
#define HSA_VEN_AMD_GRAPH_VERSION_MINOR 4

typedef struct hsa_ven_amd_graph_command_list_s {
  uint64_t handle;
} hsa_ven_amd_graph_command_list_t;

typedef enum hsa_ven_amd_graph_encoder_family_s {
  HSA_VEN_AMD_GRAPH_ENCODER_NONE = 0,
  HSA_VEN_AMD_GRAPH_ENCODER_GFX9 = 1,
  HSA_VEN_AMD_GRAPH_ENCODER_GFX10 = 2,
  HSA_VEN_AMD_GRAPH_ENCODER_GFX11 = 3,
  HSA_VEN_AMD_GRAPH_ENCODER_GFX12 = 4,
} hsa_ven_amd_graph_encoder_family_t;

typedef enum hsa_ven_amd_graph_capability_flag_s {
  HSA_VEN_AMD_GRAPH_CAPABILITY_COMPILE_SUPPORTED = 1u << 0,
  HSA_VEN_AMD_GRAPH_CAPABILITY_RUNTIME_QUALIFIED = 1u << 1,
  // The encoder accepts HSA_VEN_AMD_GRAPH_DEPENDENCY_NONE.  Callers must not send
  // that value until they have seen this bit, so an older runtime keeps seeing only
  // the two values it knows.
  HSA_VEN_AMD_GRAPH_CAPABILITY_DEPENDENCY_NONE = 1u << 2,
} hsa_ven_amd_graph_capability_flag_t;

typedef struct hsa_ven_amd_graph_capabilities_s {
  uint32_t struct_size;
  uint32_t version_major;
  uint32_t version_minor;
  hsa_ven_amd_graph_encoder_family_t encoder_family;
  uint32_t flags;
  uint32_t reserved[3];
} hsa_ven_amd_graph_capabilities_t;

// Dependency carried by the edge that precedes a dispatch.  The value at index i
// describes what the encoder must insert between dispatch i-1 and dispatch i.
typedef enum hsa_ven_amd_graph_dependency_s {
  // The previous dispatch may have written device memory this one reads or writes.
  // The encoder emits a full same-agent read-modify-write boundary.
  HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW = 0,
  // The only overlap is through vector memory operations, never a scalar load, so
  // the scalar constant cache does not have to be dropped.  Requires
  // HSA_VEN_AMD_GRAPH_KERNEL_VERIFIED_VMEM_ONLY on the consumer.
  HSA_VEN_AMD_GRAPH_DEPENDENCY_VMEM_ONLY = 1,
  // The caller asserts there is no data dependency through device memory between
  // the previous dispatch and this one.  The encoder emits nothing between them:
  // no cache invalidate and no compute-idle drain.  Requires
  // HSA_VEN_AMD_GRAPH_CAPABILITY_DEPENDENCY_NONE.
  HSA_VEN_AMD_GRAPH_DEPENDENCY_NONE = 2,
} hsa_ven_amd_graph_dependency_t;

typedef enum hsa_ven_amd_graph_kernel_flag_s {
  HSA_VEN_AMD_GRAPH_KERNEL_DYNAMIC_CALLSTACK = 1u << 0,
  HSA_VEN_AMD_GRAPH_KERNEL_VERIFIED_VMEM_ONLY = 1u << 1,
  // The consumer performs a scalar load of global memory, so a boundary in front of
  // it must also drop the scalar constant cache.  A caller that knows the consumer
  // only performs vector loads may leave this clear to let the runtime downgrade a
  // same-agent RMW edge to VMEM_ONLY itself.
  HSA_VEN_AMD_GRAPH_KERNEL_SCALAR_READS_GLOBAL = 1u << 2,
} hsa_ven_amd_graph_kernel_flag_t;

typedef enum hsa_ven_amd_graph_command_list_flag_s {
  HSA_VEN_AMD_GRAPH_COMMAND_LIST_ALLOW_UNQUALIFIED = 1u << 0,
} hsa_ven_amd_graph_command_list_flag_t;

typedef struct hsa_ven_amd_graph_command_list_desc_s {
  uint32_t struct_size;
  uint32_t reserved0;
  const hsa_kernel_dispatch_packet_t* packets;
  size_t packet_count;
  const hsa_ven_amd_graph_dependency_t* dependencies;
  size_t dependency_count;
  const uint32_t* kernel_flags;
  size_t kernel_flag_count;
  uint64_t flags;
} hsa_ven_amd_graph_command_list_desc_t;

typedef union hsa_ven_amd_graph_packet_bytes_u {
  uint8_t bytes[64];
  uint64_t qwords[8];
} hsa_ven_amd_graph_packet_bytes_t;

typedef struct hsa_ven_amd_graph_materialized_packet_s {
  hsa_ven_amd_graph_packet_bytes_t packet;
  uint32_t full_header;
  uint32_t reserved;
} hsa_ven_amd_graph_materialized_packet_t;

typedef enum hsa_ven_amd_graph_command_list_info_s {
  HSA_VEN_AMD_GRAPH_COMMAND_LIST_INFO_DISPATCH_COUNT = 0,
  HSA_VEN_AMD_GRAPH_COMMAND_LIST_INFO_DWORD_COUNT = 1,
  HSA_VEN_AMD_GRAPH_COMMAND_LIST_INFO_ENCODER_FAMILY = 2,
  HSA_VEN_AMD_GRAPH_COMMAND_LIST_INFO_MAX_PRIVATE_SEGMENT_SIZE = 3,
} hsa_ven_amd_graph_command_list_info_t;

hsa_status_t HSA_API hsa_ven_amd_graph_get_capabilities(
    hsa_agent_t agent, hsa_ven_amd_graph_capabilities_t* capabilities);

hsa_status_t HSA_API hsa_ven_amd_graph_command_list_create(
    hsa_agent_t agent, const hsa_ven_amd_graph_command_list_desc_t* desc,
    hsa_ven_amd_graph_command_list_t* command_list);

hsa_status_t HSA_API hsa_ven_amd_graph_command_list_destroy(
    hsa_ven_amd_graph_command_list_t command_list);

hsa_status_t HSA_API hsa_ven_amd_graph_command_list_get_info(
    hsa_ven_amd_graph_command_list_t command_list,
    hsa_ven_amd_graph_command_list_info_t attribute, void* value);

hsa_status_t HSA_API hsa_ven_amd_graph_command_list_materialize_packet(
    hsa_ven_amd_graph_command_list_t command_list, hsa_fence_scope_t acquire_scope,
    hsa_fence_scope_t release_scope, uint32_t barrier, hsa_signal_t completion_signal,
    hsa_ven_amd_graph_materialized_packet_t* packet);

hsa_status_t HSA_API hsa_ven_amd_graph_command_list_materialize_packet_for_queue(
    hsa_ven_amd_graph_command_list_t command_list, hsa_queue_t* queue,
    hsa_fence_scope_t acquire_scope, hsa_fence_scope_t release_scope, uint32_t barrier,
    hsa_signal_t completion_signal, hsa_ven_amd_graph_materialized_packet_t* packet);

#ifdef __cplusplus
}
#endif

#endif  // HSA_RUNTIME_INC_HSA_VEN_AMD_GRAPH_H_
