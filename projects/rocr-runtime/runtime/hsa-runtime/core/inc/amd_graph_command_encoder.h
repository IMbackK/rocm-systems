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
 * Every family lowers a same-agent RMW dependency to compute-idle plus
 * ACQUIRE_MEM.  GFX9 has no GCR_CNTL: the actions live in CP_COHER_CNTL, so the
 * pair is a PFP-level drain instead of a GCR override riding the ME dispatch
 * path, and that difference decides whether a retained list is worth building.
 *
 * Measured on a 319-dispatch graph, us per dispatch, interleaved runs, against
 * ordinary AQL on the same binary:
 *
 *   family    device          AQL    all 318 edges RMW    63 of 318 RMW
 *   gfx908    MI100          1.365              1.505            0.462
 *   gfx1100   RX 7900 XTX    2.509              0.741            0.359
 *   gfx1201   RX 9070        2.440              0.659            0.272
 *
 * Two corrections to the assumptions this file was written against.
 *
 * The boundary is not nearly free on GFX11 and GFX12.  ACQUIRE_MEM is still one
 * packet per edge there; only its action set is spelled as GCR_CNTL.  Removing
 * 255 of 318 boundaries from the same graph is worth 1.9x on gfx1100 and 2.3x
 * on gfx1201.  The retained list does still beat AQL with every edge kept,
 * which is why the caller's "replay as AQL when the classifier removed nothing"
 * policy is scoped to GFX9 and not applied here.
 *
 * The drain is not redundant with the CP's own dispatch handling, and it is not
 * a cache concern either.  A producer that spins before writing, feeding a
 * consumer that reads a buffer no kernel has ever read (so no cache line of it
 * can be stale), observes the consumer running the full spin early with no
 * boundary at all.  Measured on gfx908 and gfx1100; gfx1201 happened to stay
 * ordered.  The CP starts DISPATCH_DIRECT packets without waiting for the
 * previous one to retire, so the drain is the only thing ordering the
 * dispatches themselves, and DEPENDENCY_NONE asserts only the absence of a data
 * dependency.  It must not be read as a statement about execution order.
 *
 * Two further things about the GFX9 boundary are not obvious from the packet
 * definitions:
 *
 *   - Only SH_KCACHE_ACTION_ENA and TCL1_ACTION_ENA are needed.  Dropping
 *     SH_ICACHE_ACTION_ENA, TC_ACTION_ENA and TC_WB_ACTION_ENA keeps a
 *     512-step non-atomic dependent RMW chain exact at 4 KiB, 256 KiB and
 *     32 MiB working sets.  A consumer that provably never issues a scalar load
 *     of global memory needs TCL1_ACTION_ENA alone, which is the cheaper half:
 *     0.83 us per dispatch against 1.14 us for the pair, measured on the
 *     512-node slope.  Adding TC_ACTION_ENA makes the chain read stale data at
 *     small working sets, so it stays out.
 *   - The drain must come first.  GFX9 runs ACQUIRE_MEM in the PFP, which does
 *     not wait for the ME, so an acquire placed in front of the drain overtakes
 *     the producer, invalidates a cache the producer has not finished dirtying,
 *     and the chain silently reads stale values.  Measured on the 512-step
 *     chain at 16 KiB, 128 KiB, 256 KiB and 2 MiB: acquire-then-drain corrupts
 *     every vector, scalar and read-only flavour, drain-then-acquire is exact.
 *     Acquire-then-drain is also the cheaper of the two at 0.50 us, which is
 *     exactly why the ordering has to be stated rather than inferred from a
 *     passing benchmark.
 *
 * On the 512-node chain the resulting GFX9 boundary is 1.14 us per dependent
 * edge against 1.29 us for the same chain dispatched as ordinary AQL.
 */
inline bool ConservativeGfx9Fence() {
  static const bool enable = [] {
    const char* value = std::getenv("HSA_GRAPH_PM4_CONSERVATIVE_FENCE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enable;
}

/*
 * GFX9 dependency boundary recipe, for fault isolation.
 *
 * The CP runs ACQUIRE_MEM in the PFP and DISPATCH_DIRECT in the ME, so the PFP
 * can run the invalidate ahead of the producer it is supposed to order against.
 * The recipe selects where the invalidate sits relative to the compute-idle
 * drain, which is the only thing that keeps the PFP from starting the consumer.
 */
enum class Gfx9BoundaryRecipe {
  kDrainAcquire,      // default: CS_PARTIAL_FLUSH, then ACQUIRE_MEM
  kAcquireDrain,      // ACQUIRE_MEM, CS_PARTIAL_FLUSH (unsound, kept for bisection)
  kAcquireOnly,       // ACQUIRE_MEM
  kDrainOnly,         // CS_PARTIAL_FLUSH
  kAcquireDrainAcquire,
  kPfpSyncAcquire,    // PFP_SYNC_ME, ACQUIRE_MEM
  kNone,              // no boundary at all
};

inline Gfx9BoundaryRecipe Gfx9BoundaryRecipeFromEnv() {
  static const Gfx9BoundaryRecipe recipe = [] {
    const char* value = std::getenv("HSA_GRAPH_PM4_GFX9_BOUNDARY");
    if (value == nullptr) return Gfx9BoundaryRecipe::kDrainAcquire;
    switch (value[0]) {
      case 'd':
        // "da" drains then acquires, "d" drains only.
        return value[1] == 'a' ? Gfx9BoundaryRecipe::kDrainAcquire
                               : Gfx9BoundaryRecipe::kDrainOnly;
      case 'a':
        // "a" acquires only, "ad" acquires then drains, "ada" does both sides.
        if (value[1] != 'd') return Gfx9BoundaryRecipe::kAcquireOnly;
        return value[2] == 'a' ? Gfx9BoundaryRecipe::kAcquireDrainAcquire
                               : Gfx9BoundaryRecipe::kAcquireDrain;
      case 'p':
        return Gfx9BoundaryRecipe::kPfpSyncAcquire;
      case 'n':
        return Gfx9BoundaryRecipe::kNone;
      default:
        return Gfx9BoundaryRecipe::kDrainAcquire;
    }
  }();
  return recipe;
}

/*
 * GFX9 ACQUIRE_MEM tuning knobs, for fault isolation and cost measurement.
 * HSA_GRAPH_PM4_GFX9_COHER replaces the CP_COHER_CNTL action set;
 * HSA_GRAPH_PM4_GFX9_RANGE narrows the invalidated byte range (default all).
 */
struct Gfx9CoherOverride {
  bool present;
  uint32_t cntl;
};

// A present-but-zero setting is meaningful: an ACQUIRE_MEM with an empty action set is the
// packet without the invalidation, which separates the two halves of the boundary cost.
inline Gfx9CoherOverride Gfx9CoherCntlOverride() {
  static const Gfx9CoherOverride value = [] {
    const char* s = std::getenv("HSA_GRAPH_PM4_GFX9_COHER");
    if (s == nullptr || s[0] == '\0') return Gfx9CoherOverride{false, 0};
    return Gfx9CoherOverride{true, static_cast<uint32_t>(strtoull(s, nullptr, 0))};
  }();
  return value;
}

inline uint64_t Gfx9AcquireRangeBytes() {
  static const uint64_t value = []() -> uint64_t {
    const char* s = std::getenv("HSA_GRAPH_PM4_GFX9_RANGE");
    if (s == nullptr || s[0] == '\0') return ~(uint64_t)0;
    return static_cast<uint64_t>(strtoull(s, nullptr, 0));
  }();
  return value;
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
    // gfx1200 (12.0.0) and gfx1201 (12.0.1).  The qualification is on the
    // minor-0 family rather than on the one part that was measured: the
    // encoder keys off the family, and nothing this encoder touches - the
    // COMPUTE_PGM_RSRC* image, ACQUIRE_MEM, the dispatch packets - differs
    // between the two steppings.  The evidence is a gfx1201 board: dependent
    // RMW chain at seven working sets in vector, scalar and read-only
    // flavours, fifteen sparse-aliasing shapes with the emitted dword count
    // checked against the dependency model, graph update and node
    // enable/disable, and an LDS-saturating graph at 100 percent
    // dependency-free edges, all exact.  gfx1250 (12.5.x) stays
    // compile-unqualified: it is a different ISA revision and has not been run.
    return {HSA_VEN_AMD_GRAPH_ENCODER_GFX12, minor == 0, minor == 0};
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
      } else if (dependency == HSA_VEN_AMD_GRAPH_DEPENDENCY_NONE) {
        // The caller proved there is no data dependency through device memory
        // across this edge, so there is nothing to invalidate and nothing to
        // order against: emit the dispatch alone.  A scalar-reading consumer
        // still needs the scalar cache dropped, so refuse the downgrade when the
        // caller flagged one rather than silently keeping a stale SCache line.
        if ((kernel_flags & HSA_VEN_AMD_GRAPH_KERNEL_SCALAR_READS_GLOBAL) != 0) {
          return false;
        }
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

  // Drain first, acquire second. See the ordering note at the top of the file:
  // the CP neither serialises DISPATCH_DIRECT nor waits for the ME before it
  // runs ACQUIRE_MEM, so the invalidate has to come after a compute-idle drain
  // or it invalidates a cache the producer has not finished dirtying.
  // HSA_GRAPH_PM4_GFX9_BOUNDARY reorders or drops the two for fault isolation.
  void DependencyBoundary(uint32_t coher_cntl) {
    switch (Gfx9BoundaryRecipeFromEnv()) {
      case Gfx9BoundaryRecipe::kAcquireDrain:
        EmitAcquire(coher_cntl);
        WaitComputeIdle();
        break;
      case Gfx9BoundaryRecipe::kAcquireOnly:
        EmitAcquire(coher_cntl);
        break;
      case Gfx9BoundaryRecipe::kDrainOnly:
        WaitComputeIdle();
        break;
      case Gfx9BoundaryRecipe::kAcquireDrainAcquire:
        EmitAcquire(coher_cntl);
        WaitComputeIdle();
        EmitAcquire(coher_cntl);
        break;
      case Gfx9BoundaryRecipe::kPfpSyncAcquire:
        // PACKET3_PFP_SYNC_ME (0x60): hold the PFP until the ME has retired
        // everything before it, then invalidate.
        words_.push_back(Packet3(0x60, 1, false));
        EmitAcquire(coher_cntl);
        break;
      case Gfx9BoundaryRecipe::kNone:
        break;
      case Gfx9BoundaryRecipe::kDrainAcquire:
      default:
        WaitComputeIdle();
        EmitAcquire(coher_cntl);
        break;
    }
  }

  void EmitAcquire(uint32_t coher_cntl) {
    const Gfx9CoherOverride ovr = Gfx9CoherCntlOverride();
    if (ovr.present) {
      coher_cntl = ovr.cntl;
    }
    uint64_t size = Gfx9AcquireRangeBytes();
    if (size == ~(uint64_t)0) {
      words_.insert(words_.end(), {
          Packet3(kPacket3AcquireMem, 6, false), coher_cntl, UINT32_MAX, kCoherSizeHi, 0,
          0, kCoherPollInterval,
      });
      return;
    }
    if (size != 0) {
      size -= 1;
    }
    words_.insert(words_.end(), {
        Packet3(kPacket3AcquireMem, 6, false), coher_cntl,
        static_cast<uint32_t>(size & 0xFFFFFFFFu), static_cast<uint32_t>(size >> 32), 0, 0,
        kCoherPollInterval,
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
      } else if (dependency == HSA_VEN_AMD_GRAPH_DEPENDENCY_NONE) {
        // Nothing to invalidate and no GCR action override to apply.  A
        // scalar-reading consumer still needs the scalar cache dropped, so refuse
        // rather than leave a stale SCache line behind.
        if ((kernel_flags & HSA_VEN_AMD_GRAPH_KERNEL_SCALAR_READS_GLOBAL) != 0) {
          return false;
        }
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
      } else if (dependency == HSA_VEN_AMD_GRAPH_DEPENDENCY_NONE) {
        // Nothing to invalidate and no GCR action override to apply.  A
        // scalar-reading consumer still needs the scalar cache dropped, so refuse
        // rather than leave a stale SCache line behind.
        if ((kernel_flags & HSA_VEN_AMD_GRAPH_KERNEL_SCALAR_READS_GLOBAL) != 0) {
          return false;
        }
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
