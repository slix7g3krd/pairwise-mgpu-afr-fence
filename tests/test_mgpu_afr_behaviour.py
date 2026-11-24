"""针对题面六类 AFR 缺陷的行为回归测试。

与 ``test_mgpu_afr_fence.py`` 的契约测试互补：那里只要求入口有返回值，
这里验证返回值确实对应题目描述的错误状态。
"""

import pytest

import graphics.mgpu_afr_fence as mod
from graphics.mgpu_afr_fence import (
    AFRCoordinator,
    color_match,
    cross_adapter_fence,
    device_lost,
    frame_index_align,
    frame_partition,
    present_order,
    shared_texture,
    single_fallback,
    swapchain_rebuild,
    validation,
    vram_union,
)


# ---------------------------------------------------------------------------
# 1. 跨适配器栅栏
# ---------------------------------------------------------------------------


def test_cross_adapter_write_waits_for_producer_signal():
    out = cross_adapter_fence(producer=0, consumer=1, frame=7)
    assert out["cross_adapter"] is True
    assert out["hazard"] == "cross_write"
    assert out["ready"] is False
    assert out["blocked"] is True
    assert out["signal"] == {"adapter": 0, "value": 8, "frame": 7, "resource": None}


def test_cross_adapter_releases_once_fence_signaled():
    out = cross_adapter_fence(producer=0, consumer=1, frame=7, signaled=8)
    assert out["ready"] is True
    assert out["blocked"] is False
    assert out["reason"] == "fence_signaled"


def test_consumer_wait_alone_is_not_enough_without_producer_signal():
    # 只调用了 wait 但生产者从未 signal，仍然不能放行。
    out = cross_adapter_fence(producer=0, consumer=1, frame=7, waited=8)
    assert out["ready"] is False
    assert out["blocked"] is True


def test_same_adapter_needs_no_cross_fence():
    out = cross_adapter_fence(producer=0, consumer=0, frame=7)
    assert out["cross_adapter"] is False
    assert out["fence_required"] is False
    assert out["ready"] is True


def test_fence_timeout_and_producer_failure_are_reported():
    out = cross_adapter_fence(
        producer=0, consumer=1, frame=3, timeout_frames=2, blocked_frames=2
    )
    assert out["blocked"] is True
    assert out["timed_out"] is True

    failed = cross_adapter_fence(producer=0, consumer=1, frame=3,
                                 producer_failed=True)
    assert failed["deadlock"] is True
    assert failed["ready"] is False


# ---------------------------------------------------------------------------
# 2. 命令缓冲按帧分区
# ---------------------------------------------------------------------------


def test_each_adapter_and_slot_gets_its_own_command_buffer():
    a = frame_partition(frame=4, frames_in_flight=2, adapters=2)
    b = frame_partition(frame=5, frames_in_flight=2, adapters=2)
    assert a["adapter"] != b["adapter"]
    assert a["buffer_key"] != b["buffer_key"]
    assert a["total_buffers"] == 4
    assert a["cross_adapter_buffer_alias"] is False


def test_slot_cannot_be_reused_while_previous_frame_in_flight():
    out = frame_partition(frame=2, frames_in_flight=2, adapters=2, completed=[1])
    assert out["wait_required"] is True
    assert out["usable"] is False
    assert out["prior_frame"] == 0

    ready = frame_partition(frame=2, frames_in_flight=2, adapters=2,
                            completed=[0, 1])
    assert ready["wait_required"] is False
    assert ready["usable"] is True


def test_shared_command_buffer_between_adapters_is_flagged():
    # 两卡共享同一个命令缓冲对象，就是题面里“GPU0 写同一命令缓冲”的缺陷。
    out = frame_partition(
        frame=2,
        frames_in_flight=2,
        adapters=2,
        command_buffers={"cb:0:0": (0, 0), "cb:1:0": (0, 0)},
    )
    assert out["alias_conflict"] is True
    assert out["cross_adapter_buffer_alias"] is True
    assert out["ok"] is False


# ---------------------------------------------------------------------------
# 3. 统一显存预算
# ---------------------------------------------------------------------------


def test_shared_texture_counted_once_not_per_adapter():
    out = vram_union(
        [
            {"id": "color", "bytes": 100, "shared": True, "adapters": [0, 1]},
        ],
        budget=100,
    )
    assert out["naive_bytes"] == 200  # 各卡各算一份 = 翻倍
    assert out["unified_bytes"] == 100
    assert out["double_counted"] is True
    assert out["over_budget"] is False
    assert out["remaining"] == 0


def test_private_copies_still_count_per_adapter():
    out = vram_union(
        [
            {"id": "color", "bytes": 100, "shared": True, "adapters": [0, 1]},
            {"id": "shadow0", "bytes": 40, "shared": False, "adapters": [0]},
            {"id": "shadow1", "bytes": 40, "shared": False, "adapters": [1]},
        ],
        budget=500,
    )
    assert out["unified_bytes"] == 180
    assert out["naive_bytes"] == 280
    assert sum(out["per_adapter_naive"].values()) == out["naive_bytes"]


def test_duplicate_registration_of_same_allocation_is_merged():
    out = vram_union(
        [
            {"id": "gbuffer", "bytes": 64, "adapters": [0]},
            {"id": "gbuffer", "bytes": 64, "adapters": [1]},
        ],
    )
    assert out["unique_allocations"] == 1
    assert out["unified_bytes"] == 64


# ---------------------------------------------------------------------------
# 4. 交换链重建与帧号对齐
# ---------------------------------------------------------------------------


def test_rebuild_continues_global_frame_instead_of_resetting_to_zero():
    out = swapchain_rebuild(frame=0, last_presented=9)
    # 请求把帧号归零时必须拒绝重建，并给出正确的起始帧。
    assert out["boundary_ok"] is False
    assert out["frame_index_reset"] is False
    assert out["start_frame"] == 10
    assert out["aligned_adapter"] == 10 % mod.DEFAULT_ADAPTERS


def test_rebuild_waits_for_inflight_frames():
    out = swapchain_rebuild(
        frame=10,
        last_presented=9,
        inflight=[{"frame": 9, "complete": False}],
    )
    assert out["drained"] is False
    assert out["allowed"] is False
    assert out["drain_required"] is True
    assert out["pending_frames"] == [9]


def test_rebuild_at_boundary_keeps_adapter_parity():
    out = swapchain_rebuild(frame=10, last_presented=9)
    assert out["allowed"] is True
    assert out["start_frame"] == 10
    assert out["adapter_parity_preserved"] is True
    assert out["new_swapchain"]["base_frame"] == 10


def test_local_frame_index_maps_back_to_global_adapter():
    out = frame_index_align(local_index=0, rebuild_base=10, previous_global=9)
    assert out["global_frame"] == 10
    assert out["owner_adapter"] == 0
    assert out["local_parity_misassign"] is False
    assert out["ok"] is True


def test_local_frame_index_misassigns_adapter_after_rebuild():
    # 新交换链局部帧号 0，但全局帧是 11：直接用局部帧号会选错卡。
    out = frame_index_align(local_index=0, rebuild_base=11, previous_global=10)
    assert out["global_frame"] == 11
    assert out["owner_adapter"] == 1
    assert out["local_parity"] == 0
    assert out["local_parity_misassign"] is True
    # 有 rebuild_base 时可以重映射回正确全局帧号。
    assert out["ok"] is True
    assert out["remapped"] is True


def test_frame_index_rewind_is_rejected():
    out = frame_index_align(global_index=5, previous_global=9)
    assert out["rewind"] is True
    assert out["ok"] is False
    assert out["reason"] == "global_frame_rewind"


# ---------------------------------------------------------------------------
# 5. 色彩管线一致性
# ---------------------------------------------------------------------------


def test_identical_pipelines_match():
    out = color_match()
    assert out["match"] is True
    assert out["double_gamma"] is False
    assert out["differences"] == []


def test_srgb_plus_manual_gamma_is_double_gamma():
    out = color_match(
        reference={"transfer": "srgb", "hardware_srgb": True},
        candidate={"transfer": "srgb", "hardware_srgb": True, "shader_gamma": "srgb"},
    )
    assert out["double_gamma"] is True
    assert out["compatible"] is False
    assert out["ok"] is False
    assert out["differences"].count("shader_gamma") == 1


def test_linear_target_needs_one_shader_conversion_not_two():
    out = color_match(
        reference={"transfer": "linear", "shader_gamma": "srgb"},
        candidate={"transfer": "linear"},
    )
    # 参考卡在 shader 里做一次 sRGB 编码是对的，不是双重 gamma。
    assert out["resolved"]["reference"] == "srgb"
    assert out["double_gamma"] is False
    assert any(c["adapter"] == "candidate" and c["field"] == "transfer"
               for c in out["conversions"])


def test_range_and_color_space_mismatch_are_reported():
    out = color_match(
        reference={"transfer": "srgb", "range": "limited", "color_space": "rec709"},
        candidate={"transfer": "srgb", "range": "full", "color_space": "rec2020"},
    )
    assert "range" in out["differences"]
    assert "color_space" in out["differences"]
    assert out["target"]["range"] == "full"
    assert out["target"]["color_space"] == "rec709"


# ---------------------------------------------------------------------------
# 6. 失败回退单卡
# ---------------------------------------------------------------------------


def test_device_lost_reassigns_frames_and_bumps_fence_generation():
    out = device_lost(
        adapter=1,
        frame=7,
        generation=3,
        inflight=[{"frame": 7, "adapter": 1}, {"frame": 8, "adapter": 0}],
    )
    assert out["recoverable"] is True
    assert out["survivors"] == [0]
    assert out["fence_generation"] == 4
    assert out["rebuild_swapchain"] is True
    assert out["reassigned"][0]["to"] == 0


def test_single_fallback_disables_afr_and_shrinks_budget():
    out = single_fallback(failed=1, budget=1000, frame=8)
    assert out["afr_enabled"] is False
    assert out["mode"] == "single_adapter"
    assert out["survivor"] == 0
    assert out["budget_after"] == 500
    assert out["next_frame"] == 9


def test_no_surviving_adapter_is_unavailable_not_silently_degraded():
    out = single_fallback(failed=0, adapters=1)
    assert out["available"] is False
    assert out["ok"] is False
    assert out["mode"] == "unavailable"


# ---------------------------------------------------------------------------
# 7. 共享纹理与呈现顺序
# ---------------------------------------------------------------------------


def test_shared_texture_requires_fence_and_single_allocation():
    out = shared_texture(width=1920, height=1080)
    assert out["shared"] is True
    assert out["fence_required"] is True
    assert out["access_rule"] == "single_writer_then_fence"
    assert out["unified_bytes"] == out["bytes"]
    assert out["naive_bytes"] == out["bytes"] * 2


def test_out_of_order_arrival_is_reordered_not_presented_directly():
    out = present_order([{"frame": 1, "adapter": 1}, {"frame": 0, "adapter": 0}])
    assert out["arrival_monotonic"] is False
    assert out["ordered"] == [0, 1]
    assert out["wait_frames"] == [1]
    assert out["ok"] is True


def test_duplicate_and_missing_frames_are_flagged():
    dup = present_order([{"frame": 0}, {"frame": 0}])
    assert dup["duplicates"] == [0]
    assert dup["ok"] is False

    gap = present_order([{"frame": 0}, {"frame": 2}])
    assert gap["gaps"] == [1]
    assert gap["ok"] is False


# ---------------------------------------------------------------------------
# 8. 校验、审计与确定性
# ---------------------------------------------------------------------------


def test_validation_catches_all_six_failure_modes():
    bad = validation({
        "adapters": 2,
        "active_adapters": [0, 1],
        "failed_adapters": [1],
        "command_buffers": ["cb:shared", "cb:shared"],
        "allocations": [
            {"id": "color", "bytes": 100, "shared": True, "adapters": [0, 1]},
        ],
        "color": {
            "reference": {"transfer": "srgb", "hardware_srgb": True},
            "candidate": {"transfer": "srgb", "hardware_srgb": True,
                          "shader_gamma": "srgb"},
        },
        "present": [{"frame": 0}, {"frame": 0}],
    })
    assert bad["ok"] is False
    assert "failed_adapter_active" in bad["errors"]
    assert "command_buffer_aliasing" in bad["errors"]
    assert "double_gamma" in bad["errors"]
    assert "duplicate_present" in bad["errors"]


def test_audit_flags_unfenced_shared_texture():
    out = mod.audit({
        "adapters": 2,
        "shared_textures": [
            {"texture_id": "color", "fenced": False},
        ],
    })
    assert out["ok"] is False
    assert "shared_texture_unfenced:0" in out["problems"]


def test_audit_clean_record_passes():
    out = mod.audit({
        "adapters": 2,
        "frames_in_flight": 2,
        "present": [{"frame": 0, "adapter": 0}, {"frame": 1, "adapter": 1}],
        "color": {"reference": {"transfer": "srgb"},
                  "candidate": {"transfer": "srgb"}},
    })
    assert out["ok"] is True
    assert out["problems"] == []


def test_determinism_is_independent_of_insertion_order():
    a = mod.determinism({"x": 1, "y": [2, 3]})
    b = mod.determinism({"y": [2, 3], "x": 1})
    assert a["fingerprint"] == b["fingerprint"]
    assert a["deterministic"] is True
    assert mod.determinism({"x": 1})["fingerprint"] != b["fingerprint"]


# ---------------------------------------------------------------------------
# 9. 端到端协调器
# ---------------------------------------------------------------------------


def test_coordinator_alternates_adapters_and_never_waits_on_own_frames():
    coord = AFRCoordinator(adapters=2, frames_in_flight=2)
    adapters = []
    for frame in range(6):
        plan = coord.begin_frame(frame)
        assert plan["ok"], plan
        adapters.append(plan["adapter"])
        coord.complete_frame(frame)
    assert adapters == [0, 1, 0, 1, 0, 1]


def test_coordinator_blocks_frame_until_prior_slot_completes():
    coord = AFRCoordinator(adapters=2, frames_in_flight=2)
    coord.begin_frame(0)
    coord.begin_frame(1)
    # 帧 0/1 都没完成，帧 2 复用帧 0 的槽位，必须等待。
    plan = coord.begin_frame(2)
    assert plan["ok"] is False
    assert plan["reason"] == "wait_for_command_buffer"

    # 槽位释放后，帧 2 仍需等帧 1 的跨卡围栏；帧 1 完成后放行。
    coord.complete_frame(0)
    fence_plan = coord.begin_frame(2)
    assert fence_plan["ok"] is False
    assert fence_plan["reason"] == "wait_for_cross_adapter_fence"

    coord.complete_frame(1)
    assert coord.begin_frame(2)["ok"] is True


def test_coordinator_falls_back_to_single_adapter_after_loss():
    coord = AFRCoordinator(adapters=2, frames_in_flight=2)
    for frame in range(4):
        coord.complete_frame(frame)
    result = coord.lose_adapter(1, frame=4)
    assert result["ok"] is True
    assert coord.afr_enabled is False
    assert coord.adapter_for(5) == 0
    assert coord.state()["failed_adapters"] == [1]
