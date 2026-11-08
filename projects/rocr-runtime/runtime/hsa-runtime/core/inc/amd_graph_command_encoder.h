/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
 * SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
 */

#ifndef HSA_RUNTIME_CORE_INC_AMD_GRAPH_COMMAND_ENCODER_H_
#define HSA_RUNTIME_CORE_INC_AMD_GRAPH_COMMAND_ENCODER_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <vector>

#include "inc/hsa.h"
#include "inc/hsa_ven_amd_graph.h"

namespace rocr::graph {

/*
 * GFX9 dependency boundary width.
 *
 * The GFX11 reference lowers a same-agent RMW dependency to compute-idle plus
 * ACQUIRE_MEM, and its GCR_CNTL actions ride the ME dispatch path, so the
 * boundary is nearly free there.  GFX9 has no GCR_CNTL: the actions live in
 * CP_COHER_CNTL, and on gfx908 the pair costs roughly 4.5 us per dispatch,
 * which is 60% of throughput on a 1010-dispatch llama.cpp decode graph
 * (246 tokens/s on ordinary AQL, 140 through a retained list).
 *
 * Three things about that boundary are measurable and none of them is obvious
 * from the packet definitions:
 *
 *   - Only SH_KCACHE_ACTION_ENA and TCL1_ACTION_ENA are needed.  Dropping
 *     SH_ICACHE_ACTION_ENA, TC_ACTION_ENA and TC_WB_ACTION_ENA keeps a
 *     512-step non-atomic dependent RMW chain exact at 4 KiB, 256 KiB and
 *     32 MiB working sets and takes 140 to 227 tokens/s.  TC together with
 *     TC_WB is the degenerate combination Mesa deliberately splits into two
 *     packets, and SH_ICACHE guards against code patching rather than against
 *     another dispatch's writes.
 *   - The two words must be emitted acquire-first.  GFX9 runs ACQUIRE_MEM in
 *     the PFP, so an acquire placed after the producer dispatch overtakes it,
 *     invalidates nothing, and the chain silently reads stale data.  With the
 *     acquire first the ME retires it behind the producer in packet order, and
 *     the compute-idle drain then keeps the PFP from starting the consumer
 *     until the invalidate has landed.  Drain-first is also exact but lands at
 *     227 instead of 244, and an acquire with no drain is simply broken.
 *   - The drain is not redundant with the CP's own dispatch serialisation.  A
 *     32 MiB working set shows the CP ordering the dispatches themselves
 *     correctly with no boundary at all; the drain is there to order the
 *     invalidate, and it is unavoidable.
 *
 * Net result is 244 tokens/s against the 246 tokens/s ordinary-AQL reference.
 * HSA_GRAPH_PM4_CONSERVATIVE_FENCE=1 restores the wider CP_COHER_CNTL action
 * set for fault isolation.  It never reorders the two packets, because the
 * narrow ordering above is the correct one.
 */
inline bool ConservativeGfx9Fence() {
  static const bool enable = [] {
    const char* value = std::getenv("HSA_GRAPH_PM4_CONSERVATIVE_FENCE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enable;
}

struct GraphCommandCapability {
  hsa_ven_amd_graph_encoder_family_t family;
  bool compile_supported;
  bool runtime_qualified;
};

inline GraphCommandCapability GetGraphCommandCapability(uint32_t major, uint32_t minor,
                                                        uint32_t stepping) {
  if (major == 9) {
    return {HSA_VEN_AMD_GRAPH_ENCODER_GFX9, true, true};
  }
  if (major == 11) {
    return {HSA_VEN_AMD_GRAPH_ENCODER_GFX11, true, true};
  }
  if (major == 12) {
    return {HSA_VEN_AMD_GRAPH_ENCODER_GFX12, minor == 0, false};
  }
  return {HSA_VEN_AMD_GRAPH_ENCODER_NONE, false, false};
}

struct Gfx9KernelImage {
  uint64_t code_entry;
  uint32_t compute_pgm_rsrc1;
  uint32_t compute_pgm_rsrc2;
  uint16_t kernel_code_properties;
  uint16_t kernarg_preload_length;
};

/*
 * GFX9 (Vega, Arcturus, Aldebaran) encoder.
 *
 * SET_SH_REG offsets are relative to PACKET3_SET_SH_REG_START (0x2c00) and
 * gc_9_0 maps its compute block at MMIO 0x2000 + 0xe00, so every register this
 * encoder touches carries the same window offset as gc_10_1/gc_11_0.  The
 * packet set is what differs, and none of the GFX11 words may be replayed
 * here:
 *
 *   - COMPUTE_PGM_RSRC3 does not exist.  Window offset 0x228 is
 *     COMPUTE_STATIC_THREAD_MGMT_SE7 on GFX9, so writing the GFX11 RSRC3 slot
 *     would mask shader engines.  Wave32 mode is absent as well; a kernel
 *     declaring it is rejected rather than silently run as wave64.
 *   - ACQUIRE_MEM carries no GCR_CNTL.  Coherency actions live in the
 *     CP_COHER_CNTL bits and CP_COHER_SIZE_HI is 24 bits wide, matching
 *     gfx_v9_0_emit_mem_sync().
 *   - COMPUTE_DISPATCH_INITIATOR has no CS_W32_EN.  A plain compute dispatch is
 *     COMPUTE_SHADER_EN plus FORCE_START_AT_000 so COMPUTE_START_* left behind
 *     by another resident cannot shrink the grid.
 *   - The CP_COHER_CNTL acquire is a synchronous CP-side operation rather than
 *     a GCR action folded into the ME dispatch path, so its action set and its
 *     ordering against the compute-idle drain both matter for throughput.  See
 *     ConservativeGfx9Fence().
 */
class Gfx9CommandEncoder {
 public:
  Gfx9CommandEncoder() { AcquireSystem(); }

  bool Append(const hsa_kernel_dispatch_packet_t& packet, const Gfx9KernelImage& image,
              uint32_t kernel_flags, hsa_ven_amd_graph_dependency_t dependency) {
    if ((kernel_flags & HSA_VEN_AMD_GRAPH_KERNEL_DYNAMIC_CALLSTACK) != 0 ||
        (image.kernel_code_properties & ~kSupportedKernelProperties) != 0 ||
        (packet.private_segment_size != 0 &&
         (image.kernel_code_properties & kEnableSgprPrivateSegmentBuffer) != 0) ||
        image.kernarg_preload_length != 0 || image.code_entry == 0 ||
        (image.code_entry & 0xff) != 0) {
      return false;
    }
    if (packet.workgroup_size_x == 0 || packet.workgroup_size_y == 0 ||
        packet.workgroup_size_z == 0 ||
        packet.grid_size_x % packet.workgroup_size_x != 0 ||
        packet.grid_size_y % packet.workgroup_size_y != 0 ||
        packet.grid_size_z % packet.workgroup_size_z != 0) {
      return false;
    }

    const uint32_t lds_blocks = packet.group_segment_size / kLdsGranule +
                                (packet.group_segment_size % kLdsGranule != 0);
    if (lds_blocks > (kLdsSizeMask >> kLdsSizeShift)) {
      return false;
    }
    const uint32_t rsrc2 =
        (image.compute_pgm_rsrc2 & ~kLdsSizeMask) | (lds_blocks << kLdsSizeShift);

    if (dispatch_count_ != 0) {
      if (dependency == HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW) {
        DependencyBoundary(DependencyCoherCntl());
      } else if (dependency == HSA_VEN_AMD_GRAPH_DEPENDENCY_VMEM_ONLY &&
                 (kernel_flags & HSA_VEN_AMD_GRAPH_KERNEL_VERIFIED_VMEM_ONLY) != 0) {
        DependencyBoundary(kCoherCntlVmem);
      } else {
        return false;
      }
    }

    const uint32_t pgm[] = {
        static_cast<uint32_t>(image.code_entry >> 8),
        static_cast<uint32_t>(image.code_entry >> 40),
    };
    SetShRegs(kComputePgmLo, pgm, 2);
    const uint32_t rsrc[] = {image.compute_pgm_rsrc1, rsrc2};
    SetShRegs(kComputePgmRsrc1, rsrc, 2);
    const uint32_t zero = 0;
    const size_t tmpring_words_before = words_.size();
    SetShRegs(kComputeTmpRingSize, &zero, 1);
    if (tmpring_patch_dword_ == kNoPatchDword && words_.size() != tmpring_words_before) {
      tmpring_patch_dword_ = words_.size() - 1;
    }
    const uint32_t workgroup[] = {packet.workgroup_size_x, packet.workgroup_size_y,
                                  packet.workgroup_size_z};
    SetShRegs(kComputeNumThreadX, workgroup, 3);
    SetShRegs(kComputeResourceLimits, &zero, 1);

    uint32_t user_sgprs[6] = {};
    size_t user_sgpr_count = 0;
    if ((image.kernel_code_properties & kEnableSgprPrivateSegmentBuffer) != 0) {
      user_sgpr_count += 4;
    }
    if ((image.kernel_code_properties & kEnableSgprKernargSegmentPtr) != 0) {
      if (packet.kernarg_address == nullptr) {
        return false;
      }
      const uint64_t kernarg = reinterpret_cast<uint64_t>(packet.kernarg_address);
      user_sgprs[user_sgpr_count++] = static_cast<uint32_t>(kernarg);
      user_sgprs[user_sgpr_count++] = static_cast<uint32_t>(kernarg >> 32);
    }
    if (user_sgpr_count != 0) {
      SetShRegs(kComputeUserData0, user_sgprs, user_sgpr_count);
    }

    const uint32_t workgroups[] = {
        packet.grid_size_x / packet.workgroup_size_x,
        packet.grid_size_y / packet.workgroup_size_y,
        packet.grid_size_z / packet.workgroup_size_z,
    };
    words_.push_back(Packet3(kPacket3DispatchDirect, 4, true));
    words_.insert(words_.end(), workgroups, workgroups + 3);
    words_.push_back(kDispatchInitiator);
    ++dispatch_count_;
    return true;
  }

  void Finish() { WaitComputeIdle(); }
  const std::vector<uint32_t>& words() const { return words_; }
  size_t dispatch_count() const { return dispatch_count_; }
  size_t tmpring_patch_dword() const { return tmpring_patch_dword_; }

 private:
  static constexpr uint32_t kPacket3SetShReg = 0x76;
  static constexpr uint32_t kPacket3DispatchDirect = 0x15;
  static constexpr uint32_t kPacket3EventWrite = 0x46;
  static constexpr uint32_t kPacket3AcquireMem = 0x58;
  static constexpr uint32_t kComputeNumThreadX = 0x207;
  static constexpr uint32_t kComputePgmLo = 0x20c;
  static constexpr uint32_t kComputePgmRsrc1 = 0x212;
  static constexpr uint32_t kComputeResourceLimits = 0x215;
  static constexpr uint32_t kComputeTmpRingSize = 0x218;
  static constexpr uint32_t kComputeUserData0 = 0x240;
  static constexpr uint32_t kLdsSizeMask = 0x00ff8000;
  static constexpr uint32_t kLdsSizeShift = 15;
  static constexpr uint32_t kLdsGranule = 512;
  static constexpr uint16_t kEnableSgprPrivateSegmentBuffer = 1 << 0;
  static constexpr uint16_t kEnableSgprKernargSegmentPtr = 1 << 3;
  // Wavefront-size-32 (bit 10) is deliberately unsupported: GFX9 has no
  // CS_W32_EN, so a wave32 kernel must fail closed instead of running wide.
  static constexpr uint16_t kSupportedKernelProperties =
      kEnableSgprPrivateSegmentBuffer | kEnableSgprKernargSegmentPtr;
  static constexpr uint32_t kDispatchInitiator = (1u << 0) | (1u << 2);
  // CP_COHER_CNTL actions a same-agent dependency boundary needs on GFX9:
  // SH_KCACHE_ACTION_ENA drops the scalar constant cache and
  // TCL1_ACTION_ENA drops the per-CU vector read cache.  See
  // ConservativeGfx9Fence() for what the qualification sweep ruled out.
  static constexpr uint32_t kCoherCntlDependency = (1u << 27) | (1u << 22);
  // Narrowed boundary for a verified VMEM-only consumer: the vector read cache
  // without the scalar constant cache action.
  static constexpr uint32_t kCoherCntlVmem = (1u << 22);
  // Debug-gate action set: SH_ICACHE | SH_KCACHE | TC | TCL1 | TC_WB.
  static constexpr uint32_t kCoherCntlFull =
      (1u << 29) | (1u << 27) | (1u << 23) | (1u << 22) | (1u << 18);
  static constexpr uint32_t kCoherSizeHi = 0x00ffffff;
  static constexpr uint32_t kCoherPollInterval = 10;
  // EVENT_TYPE CS_PARTIAL_FLUSH with EVENT_INDEX compute.
  static constexpr uint32_t kEventCsPartialFlush = (7u << 0) | (4u << 8);

  static uint32_t DependencyCoherCntl() {
    static const uint32_t mask = ConservativeGfx9Fence() ? kCoherCntlFull
                                                         : kCoherCntlDependency;
    return mask;
  }

  static uint32_t Packet3(uint32_t opcode, uint32_t body_dwords, bool compute) {
    return (3u << 30) | ((body_dwords - 1) << 16) | (opcode << 8) |
           (compute ? 1u << 1 : 0);
  }

  void AcquireSystem() { EmitAcquire(DependencyCoherCntl()); }

  void WaitComputeIdle() {
    words_.push_back(Packet3(kPacket3EventWrite, 1, false));
    words_.push_back(kEventCsPartialFlush);
  }

  // Acquire first, drain second: see ConservativeGfx9Fence().
  void DependencyBoundary(uint32_t coher_cntl) {
    EmitAcquire(coher_cntl);
    WaitComputeIdle();
  }

  void EmitAcquire(uint32_t coher_cntl) {
    words_.insert(words_.end(), {
        Packet3(kPacket3AcquireMem, 6, false), coher_cntl, UINT32_MAX, kCoherSizeHi, 0,
        0, kCoherPollInterval,
    });
  }

  void SetShRegs(uint32_t first, const uint32_t* values, size_t count) {
    size_t i = 0;
    while (i < count) {
      while (i < count) {
        const uint32_t reg = first + static_cast<uint32_t>(i);
        const auto it = register_state_.find(reg);
        if (it == register_state_.end() || it->second != values[i]) {
          break;
        }
        ++i;
      }
      if (i == count) {
        break;
      }
      const size_t run_start = i;
      while (i < count) {
        const uint32_t reg = first + static_cast<uint32_t>(i);
        const auto it = register_state_.find(reg);
        if (it != register_state_.end() && it->second == values[i]) {
          break;
        }
        register_state_[reg] = values[i];
        ++i;
      }
      const size_t run_count = i - run_start;
      words_.push_back(Packet3(kPacket3SetShReg,
                               1 + static_cast<uint32_t>(run_count), true));
      words_.push_back(first + static_cast<uint32_t>(run_start));
      words_.insert(words_.end(), values + run_start, values + run_start + run_count);
    }
  }

  std::vector<uint32_t> words_;
  std::map<uint32_t, uint32_t> register_state_;
  size_t dispatch_count_ = 0;
  static constexpr size_t kNoPatchDword = static_cast<size_t>(-1);
  size_t tmpring_patch_dword_ = kNoPatchDword;
};

struct Gfx11KernelImage {
  uint64_t code_entry;
  uint32_t compute_pgm_rsrc1;
  uint32_t compute_pgm_rsrc2;
  uint32_t compute_pgm_rsrc3;
  uint16_t kernel_code_properties;
  uint16_t kernarg_preload_length;
};

class Gfx11CommandEncoder {
 public:
  Gfx11CommandEncoder() { AcquireSystem(); }

  bool Append(const hsa_kernel_dispatch_packet_t& packet, const Gfx11KernelImage& image,
              uint32_t kernel_flags, hsa_ven_amd_graph_dependency_t dependency) {
    if ((kernel_flags & HSA_VEN_AMD_GRAPH_KERNEL_DYNAMIC_CALLSTACK) != 0 ||
        (image.kernel_code_properties & ~kSupportedKernelProperties) != 0 ||
        (packet.private_segment_size != 0 &&
         (image.kernel_code_properties & kEnableSgprPrivateSegmentBuffer) != 0) ||
        image.kernarg_preload_length != 0 || image.code_entry == 0 ||
        (image.code_entry & 0xff) != 0) {
      return false;
    }
    if (packet.workgroup_size_x == 0 || packet.workgroup_size_y == 0 ||
        packet.workgroup_size_z == 0 ||
        packet.grid_size_x % packet.workgroup_size_x != 0 ||
        packet.grid_size_y % packet.workgroup_size_y != 0 ||
        packet.grid_size_z % packet.workgroup_size_z != 0) {
      return false;
    }

    const uint32_t lds_blocks = packet.group_segment_size / kLdsGranule +
                                (packet.group_segment_size % kLdsGranule != 0);
    if (lds_blocks > (kLdsSizeMask >> kLdsSizeShift)) {
      return false;
    }
    const uint32_t rsrc2 =
        (image.compute_pgm_rsrc2 & ~kLdsSizeMask) | (lds_blocks << kLdsSizeShift);

    if (dispatch_count_ != 0) {
      if (dependency == HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW) {
        DependencyRmwSameAgent();
      } else if (dependency == HSA_VEN_AMD_GRAPH_DEPENDENCY_VMEM_ONLY &&
                 (kernel_flags & HSA_VEN_AMD_GRAPH_KERNEL_VERIFIED_VMEM_ONLY) != 0) {
        DependencyRmwVmem();
      } else {
        return false;
      }
    }

    const uint32_t pgm[] = {
        static_cast<uint32_t>(image.code_entry >> 8),
        static_cast<uint32_t>(image.code_entry >> 40),
    };
    SetShRegs(kComputePgmLo, pgm, 2);
    const uint32_t rsrc[] = {image.compute_pgm_rsrc1, rsrc2};
    SetShRegs(kComputePgmRsrc1, rsrc, 2);
    SetShRegs(kComputePgmRsrc3, &image.compute_pgm_rsrc3, 1);
    const uint32_t zero = 0;
    const size_t tmpring_words_before = words_.size();
    SetShRegs(kComputeTmpRingSize, &zero, 1);
    if (tmpring_patch_dword_ == kNoPatchDword && words_.size() != tmpring_words_before) {
      tmpring_patch_dword_ = words_.size() - 1;
    }
    const uint32_t workgroup[] = {packet.workgroup_size_x, packet.workgroup_size_y,
                                  packet.workgroup_size_z};
    SetShRegs(kComputeNumThreadX, workgroup, 3);
    SetShRegs(kComputeResourceLimits, &zero, 1);

    uint32_t user_sgprs[6] = {};
    size_t user_sgpr_count = 0;
    if ((image.kernel_code_properties & kEnableSgprPrivateSegmentBuffer) != 0) {
      user_sgpr_count += 4;
    }
    if ((image.kernel_code_properties & kEnableSgprKernargSegmentPtr) != 0) {
      if (packet.kernarg_address == nullptr) {
        return false;
      }
      const uint64_t kernarg = reinterpret_cast<uint64_t>(packet.kernarg_address);
      user_sgprs[user_sgpr_count++] = static_cast<uint32_t>(kernarg);
      user_sgprs[user_sgpr_count++] = static_cast<uint32_t>(kernarg >> 32);
    }
    if (user_sgpr_count != 0) {
      SetShRegs(kComputeUserData0, user_sgprs, user_sgpr_count);
    }

    const uint32_t workgroups[] = {
        packet.grid_size_x / packet.workgroup_size_x,
        packet.grid_size_y / packet.workgroup_size_y,
        packet.grid_size_z / packet.workgroup_size_z,
    };
    const uint32_t wave_mode =
        (image.kernel_code_properties & kEnableWavefrontSize32) != 0 ? 1u << 15 : 0;
    const uint32_t initiator = (1u << 0) | (1u << 2) | (1u << 3) | wave_mode;
    words_.push_back(Packet3(kPacket3DispatchDirect, 4, true));
    words_.insert(words_.end(), workgroups, workgroups + 3);
    words_.push_back(initiator);
    ++dispatch_count_;
    return true;
  }

  void Finish() { WaitComputeIdle(); }
  const std::vector<uint32_t>& words() const { return words_; }
  size_t dispatch_count() const { return dispatch_count_; }
  size_t tmpring_patch_dword() const { return tmpring_patch_dword_; }

 private:
  static constexpr uint32_t kPacket3SetShReg = 0x76;
  static constexpr uint32_t kPacket3DispatchDirect = 0x15;
  static constexpr uint32_t kPacket3EventWrite = 0x46;
  static constexpr uint32_t kPacket3AcquireMem = 0x58;
  static constexpr uint32_t kComputeNumThreadX = 0x207;
  static constexpr uint32_t kComputePgmLo = 0x20c;
  static constexpr uint32_t kComputePgmRsrc1 = 0x212;
  static constexpr uint32_t kComputeResourceLimits = 0x215;
  static constexpr uint32_t kComputeTmpRingSize = 0x218;
  static constexpr uint32_t kComputePgmRsrc3 = 0x228;
  static constexpr uint32_t kComputeUserData0 = 0x240;
  static constexpr uint32_t kLdsSizeMask = 0x00ff8000;
  static constexpr uint32_t kLdsSizeShift = 15;
  static constexpr uint32_t kLdsGranule = 512;
  static constexpr uint16_t kEnableSgprPrivateSegmentBuffer = 1 << 0;
  static constexpr uint16_t kEnableSgprKernargSegmentPtr = 1 << 3;
  static constexpr uint16_t kEnableWavefrontSize32 = 1 << 10;
  static constexpr uint16_t kSupportedKernelProperties =
      kEnableSgprPrivateSegmentBuffer | kEnableSgprKernargSegmentPtr |
      kEnableWavefrontSize32;

  static uint32_t Packet3(uint32_t opcode, uint32_t body_dwords, bool compute) {
    return (3u << 30) | ((body_dwords - 1) << 16) | (opcode << 8) |
           (compute ? 1u << 1 : 0);
  }

  void AcquireSystem() {
    words_.insert(words_.end(), {
        Packet3(kPacket3AcquireMem, 7, false), 0, UINT32_MAX, 0xff, 0, 0, 4,
        (1u << 16) | (1u << 15) | (1u << 14) | (1u << 9) | (1u << 8) |
            (1u << 7) | (1u << 6) | (1u << 5) | (1u << 4) | 1u,
    });
  }

  void WaitComputeIdle() {
    words_.push_back(Packet3(kPacket3EventWrite, 1, false));
    words_.push_back(0x407);
  }

  void DependencyRmwSameAgent() {
    WaitComputeIdle();
    EmitAcquire(0x00380);
  }

  void DependencyRmwVmem() {
    WaitComputeIdle();
    EmitAcquire(0x00300);
  }

  void EmitAcquire(uint32_t gcr_cntl) {
    words_.insert(words_.end(), {
        Packet3(kPacket3AcquireMem, 7, false), 0, UINT32_MAX, 0x00ffffff, 0, 0, 10,
        gcr_cntl,
    });
  }

  void SetShRegs(uint32_t first, const uint32_t* values, size_t count) {
    size_t i = 0;
    while (i < count) {
      while (i < count) {
        const uint32_t reg = first + static_cast<uint32_t>(i);
        const auto it = register_state_.find(reg);
        if (it == register_state_.end() || it->second != values[i]) {
          break;
        }
        ++i;
      }
      if (i == count) {
        break;
      }
      const size_t run_start = i;
      while (i < count) {
        const uint32_t reg = first + static_cast<uint32_t>(i);
        const auto it = register_state_.find(reg);
        if (it != register_state_.end() && it->second == values[i]) {
          break;
        }
        register_state_[reg] = values[i];
        ++i;
      }
      const size_t run_count = i - run_start;
      words_.push_back(Packet3(kPacket3SetShReg,
                               1 + static_cast<uint32_t>(run_count), true));
      words_.push_back(first + static_cast<uint32_t>(run_start));
      words_.insert(words_.end(), values + run_start, values + run_start + run_count);
    }
  }

  std::vector<uint32_t> words_;
  std::map<uint32_t, uint32_t> register_state_;
  size_t dispatch_count_ = 0;
  static constexpr size_t kNoPatchDword = static_cast<size_t>(-1);
  size_t tmpring_patch_dword_ = kNoPatchDword;
};

struct Gfx12KernelImage {
  uint64_t code_entry;
  uint32_t compute_pgm_rsrc1;
  uint32_t compute_pgm_rsrc2;
  uint32_t compute_pgm_rsrc3;
  uint16_t kernel_code_properties;
  uint16_t kernarg_preload_length;
};

class Gfx12CommandEncoder {
 public:
  Gfx12CommandEncoder() { AcquireInterNode(); }

  bool Append(const hsa_kernel_dispatch_packet_t& packet, const Gfx12KernelImage& image,
              uint32_t kernel_flags, hsa_ven_amd_graph_dependency_t dependency) {
    if ((kernel_flags & HSA_VEN_AMD_GRAPH_KERNEL_DYNAMIC_CALLSTACK) != 0 ||
        (image.kernel_code_properties & ~kSupportedKernelProperties) != 0 ||
        image.kernarg_preload_length != 0 || image.code_entry == 0 ||
        (image.code_entry & 0xff) != 0) {
      return false;
    }
    if (packet.workgroup_size_x == 0 || packet.workgroup_size_y == 0 ||
        packet.workgroup_size_z == 0) {
      return false;
    }

    const uint32_t lds_blocks = packet.group_segment_size / kLdsGranule +
                                (packet.group_segment_size % kLdsGranule != 0);
    if (lds_blocks > (kLdsSizeMask >> kLdsSizeShift)) {
      return false;
    }
    const uint32_t rsrc2 =
        (image.compute_pgm_rsrc2 & ~kLdsSizeMask) | (lds_blocks << kLdsSizeShift);

    if (dispatch_count_ != 0) {
      if (dependency == HSA_VEN_AMD_GRAPH_DEPENDENCY_SAME_AGENT_RMW) {
        DependencyRmwSameAgent();
      } else if (dependency == HSA_VEN_AMD_GRAPH_DEPENDENCY_VMEM_ONLY &&
                 (kernel_flags & HSA_VEN_AMD_GRAPH_KERNEL_VERIFIED_VMEM_ONLY) != 0) {
        DependencyRmwVmem();
      } else {
        return false;
      }
    }

    const uint32_t pgm[] = {
        static_cast<uint32_t>(image.code_entry >> 8),
        static_cast<uint32_t>(image.code_entry >> 40),
    };
    SetShRegs(kComputePgmLo, pgm, 2);
    const uint32_t rsrc[] = {image.compute_pgm_rsrc1, rsrc2};
    SetShRegs(kComputePgmRsrc1, rsrc, 2);
    SetShRegs(kComputePgmRsrc3, &image.compute_pgm_rsrc3, 1);
    const uint32_t zero = 0;
    SetShRegs(kComputeTmpRingSize, &zero, 1);
    const uint32_t workgroup[] = {packet.workgroup_size_x, packet.workgroup_size_y,
                                  packet.workgroup_size_z};
    SetShRegs(kComputeNumThreadX, workgroup, 3);
    const uint32_t resource_limits = 0x3ff;
    SetShRegs(kComputeResourceLimits, &resource_limits, 1);

    if ((image.kernel_code_properties & kEnableSgprKernargSegmentPtr) != 0) {
      if (packet.kernarg_address == nullptr) {
        return false;
      }
      const uint64_t kernarg = reinterpret_cast<uint64_t>(packet.kernarg_address);
      const uint32_t user_sgprs[] = {static_cast<uint32_t>(kernarg),
                                     static_cast<uint32_t>(kernarg >> 32)};
      SetShRegs(kComputeUserData0, user_sgprs, 2);
    }

    FlushShRegs();

    const uint32_t dimensions[] = {packet.grid_size_x, packet.grid_size_y,
                                   packet.grid_size_z};
    const uint32_t wave_mode =
        (image.kernel_code_properties & kEnableWavefrontSize32) != 0 ? 1u << 15 : 0;
    const uint32_t initiator = (1u << 0) | (1u << 2) | (1u << 5) | wave_mode;
    words_.push_back(Packet3(kPacket3DispatchDirect, 4, true));
    words_.insert(words_.end(), dimensions, dimensions + 3);
    words_.push_back(initiator);
    ++dispatch_count_;
    return true;
  }

  void Finish() { WaitComputeIdle(); }
  const std::vector<uint32_t>& words() const { return words_; }
  size_t dispatch_count() const { return dispatch_count_; }
  size_t tmpring_patch_dword() const { return tmpring_patch_dword_; }

 private:
  static constexpr uint32_t kPacket3SetShRegPairs = 0xba;
  static constexpr uint32_t kPacket3DispatchDirect = 0x15;
  static constexpr uint32_t kPacket3EventWrite = 0x46;
  static constexpr uint32_t kPacket3AcquireMem = 0x58;
  static constexpr uint32_t kComputeNumThreadX = 0x207;
  static constexpr uint32_t kComputePgmLo = 0x20c;
  static constexpr uint32_t kComputePgmRsrc1 = 0x212;
  static constexpr uint32_t kComputeResourceLimits = 0x215;
  static constexpr uint32_t kComputeTmpRingSize = 0x218;
  static constexpr uint32_t kComputePgmRsrc3 = 0x228;
  static constexpr uint32_t kComputeUserData0 = 0x240;
  static constexpr uint32_t kLdsSizeMask = 0x00ff8000;
  static constexpr uint32_t kLdsSizeShift = 15;
  static constexpr uint32_t kLdsGranule = 512;
  static constexpr uint16_t kEnableSgprKernargSegmentPtr = 1 << 3;
  static constexpr uint16_t kEnableWavefrontSize32 = 1 << 10;
  static constexpr uint16_t kSupportedKernelProperties =
      kEnableSgprKernargSegmentPtr | kEnableWavefrontSize32;

  static uint32_t Packet3(uint32_t opcode, uint32_t body_dwords, bool compute) {
    return (3u << 30) | ((body_dwords - 1) << 16) | (opcode << 8) |
           (compute ? 1u << 1 : 0);
  }

  void AcquireInterNode() { EmitAcquire(0xff, 4, 0x10180); }

  void WaitComputeIdle() {
    words_.push_back(Packet3(kPacket3EventWrite, 1, false));
    words_.push_back(0x407);
  }

  void DependencyRmwSameAgent() {
    WaitComputeIdle();
    EmitAcquire(0x00ffffff, 10, 0x00380);
  }

  void DependencyRmwVmem() {
    WaitComputeIdle();
    EmitAcquire(0x00ffffff, 10, 0x00300);
  }

  void EmitAcquire(uint32_t size_hi, uint32_t poll_interval, uint32_t gcr_cntl) {
    words_.insert(words_.end(), {
        Packet3(kPacket3AcquireMem, 7, false), 0, UINT32_MAX, size_hi, 0, 0,
        poll_interval, gcr_cntl,
    });
  }

  void SetShRegs(uint32_t first, const uint32_t* values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      const uint32_t reg = first + static_cast<uint32_t>(i);
      const auto it = register_state_.find(reg);
      if (it == register_state_.end() || it->second != values[i]) {
        register_state_[reg] = values[i];
        pending_registers_.push_back(reg);
        pending_registers_.push_back(values[i]);
      }
    }
  }

  void FlushShRegs() {
    if (pending_registers_.empty()) {
      return;
    }
    const size_t header = words_.size();
    words_.push_back(Packet3(kPacket3SetShRegPairs,
                             static_cast<uint32_t>(pending_registers_.size()), true));
    words_.insert(words_.end(), pending_registers_.begin(), pending_registers_.end());
    if (tmpring_patch_dword_ == kNoPatchDword) {
      for (size_t i = 0; i < pending_registers_.size(); i += 2) {
        if (pending_registers_[i] == kComputeTmpRingSize) {
          tmpring_patch_dword_ = header + i + 2;
          break;
        }
      }
    }
    pending_registers_.clear();
  }

  std::vector<uint32_t> words_;
  std::vector<uint32_t> pending_registers_;
  std::map<uint32_t, uint32_t> register_state_;
  size_t dispatch_count_ = 0;
  static constexpr size_t kNoPatchDword = static_cast<size_t>(-1);
  size_t tmpring_patch_dword_ = kNoPatchDword;
};

}  // namespace rocr::graph

#endif  // HSA_RUNTIME_CORE_INC_AMD_GRAPH_COMMAND_ENCODER_H_
