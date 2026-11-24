"""多 GPU 交替帧渲染（AFR）的跨适配器同步与资源管理。

线上问题的六个根因与对应修复
------------------------------

1. **跨适配器同步**（``cross_adapter_fence`` / ``shared_texture``）
   两个适配器共享同一张渲染目标，但只有本卡内部的围栏，没有跨卡栅栏。
   GPU1 还在写第 N 帧时 GPU0 已经开始往同一块共享纹理写第 N+1 帧，画面
   出现撕裂和错帧。修复方式：每次跨卡访问都要在 *生产者* 上 signal、在
   *消费者* 上 wait 同一个围栏值；写-写、读-写都视为 hazard，未满足时
   阻塞而不是放行。

2. **命令缓冲按帧分区**（``frame_partition``）
   所有帧复用一个命令缓冲环形队列，帧 N 的录制会覆盖仍在 GPU 上执行的
   帧 N-K。修复方式：每个 (帧槽位, 适配器) 拥有独立命令缓冲；槽位按
   ``frame % frames_in_flight`` 轮转，只有占用该槽位的旧帧完成围栏后
   才能复用。

3. **统一显存预算**（``vram_union``）
   两块卡各自统计显存，跨适配器共享纹理在两边各算一次，聚合预算凭空
   翻倍。修复方式：共享分配按分配 ID 去重，只计一次；真正复制到每个
   适配器的资源才按副本数累计；同时给出“朴素相加”与“去重后”的差值，
   便于审计。

4. **交换链重建的帧号对齐**（``swapchain_rebuild`` / ``frame_index_align``）
   窗口拖到另一块屏时交换链被重建，帧号从 0 重新开始；AFR 的奇偶帧
   分配随之中断，同一帧被两块卡各渲染一次或直接被跳过。修复方式：帧号
   全局单调，重建只在帧边界、排空在途帧后进行，新交换链的起始帧号继续
   自 ``last_presented + 1``，并按旧基址推算出的适配器奇偶性对齐。

5. **色彩管线一致性**（``color_match``）
   一块卡用硬件 sRGB 转换、另一块在 shader 里手动 gamma，或两边交换链
   格式/色域/量化范围不同，同一画面在两卡间出现轻微色差。修复方式：
   比较完整的输出管线，选出共同目标（位深、色域、传递函数、full range），
   明确哪一步转换只在 shader 里做一次，禁止隐式双重 gamma。

6. **失败回退单卡**（``single_fallback`` / ``device_lost``）
   任一适配器设备丢失或围栏超时后若继续 AFR，会永久卡住。修复方式：
   把在途帧改派到存活适配器、围栏代际整体提升、预算收缩为单卡，并重建
   交换链；无法恢复时明确返回不可用而不是静默产出错帧。

其余入口
--------
``present_order`` 保证两个适配器的帧按帧号顺序呈现；``validation`` /
``audit`` / ``metrics`` 负责观测与验收；``fallback`` 在多种降级策略中做
选择；``determinism`` 给出与插入顺序无关的状态指纹，方便回归测试。

所有公开函数都不修改入参，返回可 JSON 序列化的字典；传 ``None``、标量或
任意字典都能得到有意义的判定，便于契约测试和审计工具复用。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = [
    "DEFAULT_ADAPTERS",
    "DEFAULT_FRAMES_IN_FLIGHT",
    "DEFAULT_BUDGET_BYTES",
    "TRANSFER_FUNCTIONS",
    "COLOR_SPACE_RANK",
    "DEFAULT_COLOR_PIPELINE",
    "cross_adapter_fence",
    "frame_partition",
    "vram_union",
    "swapchain_rebuild",
    "frame_index_align",
    "color_match",
    "single_fallback",
    "shared_texture",
    "present_order",
    "metrics",
    "device_lost",
    "validation",
    "audit",
    "fallback",
    "determinism",
    "AFRCoordinator",
]

DEFAULT_ADAPTERS = 2
DEFAULT_FRAMES_IN_FLIGHT = 2
DEFAULT_BUDGET_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB 统一预算

TRANSFER_FUNCTIONS = ("linear", "srgb", "gamma22", "pq", "hlg")
# 色域按从窄到宽排序；取交集时选择较窄的一方，避免超出显示器范围。
COLOR_SPACE_RANK = {
    "srgb": 1,
    "rec709": 1,
    "display_p3": 2,
    "dcip3": 2,
    "rec2020": 3,
    "scrgb": 3,
}
DEFAULT_COLOR_PIPELINE = {
    "transfer": "srgb",
    "color_space": "srgb",
    "bit_depth": 8,
    "range": "full",
    "format": "r8g8b8a8_unorm_srgb",
    "hardware_srgb": False,
    "shader_gamma": None,
}


# ---------------------------------------------------------------------------
# 通用解析辅助
# ---------------------------------------------------------------------------


def _mapping(value: Any) -> dict[str, Any]:
    """把 Mapping 复制成普通 dict；其它类型返回空 dict。"""

    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_int(value: Any, default: int) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        if isinstance(value, bool):
            return float(value)
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "y")
    return bool(value)


def _pick(opts: Mapping[str, Any], value: Any, *names: str, default: Any = None) -> Any:
    """显式参数优先；否则依次从选项字典里取第一个存在的键。"""

    if value is not None:
        return value
    for name in names:
        if name in opts:
            return opts[name]
    return default


def _normalize_frames(value: Any) -> list[dict[str, Any]]:
    """把帧列表规范化为 ``{"frame", "adapter", "complete"}``。"""

    if value is None:
        return []
    if isinstance(value, Mapping):
        value = [value]
    elif isinstance(value, (int, float, str)):
        value = [value]

    frames: list[dict[str, Any]] = []
    try:
        iterator: Iterable[Any] = value
    except TypeError:
        return frames

    for item in iterator:
        if isinstance(item, Mapping):
            raw = item.get("frame", item.get("index"))
            frame = _as_int(raw, -1)
            if frame < 0:
                continue
            adapter = item.get("adapter")
            frames.append(
                {
                    "frame": frame,
                    "adapter": _as_int(adapter, frame % DEFAULT_ADAPTERS)
                    if adapter is not None
                    else None,
                    "complete": _as_bool(
                        item.get("complete", item.get("done", False))
                    ),
                }
            )
        else:
            frame = _as_int(item, -1)
            if frame < 0:
                continue
            frames.append({"frame": frame, "adapter": None, "complete": False})
    return frames


def _align_up(value: int, alignment: int) -> int:
    alignment = max(1, int(alignment))
    return ((max(0, int(value)) + alignment - 1) // alignment) * alignment


# ---------------------------------------------------------------------------
# 1. 跨适配器栅栏
# ---------------------------------------------------------------------------


def cross_adapter_fence(
    spec: Any = None,
    *args: Any,
    producer: Any = None,
    consumer: Any = None,
    frame: Any = None,
    access: Any = None,
    fence_value: Any = None,
    signaled: Any = None,
    waited: Any = None,
    timeout_frames: Any = None,
    blocked_frames: Any = None,
    producer_failed: Any = None,
    resource: Any = None,
) -> dict[str, Any]:
    """计算一次跨适配器访问所需的围栏。

    生产者在自己的队列上 ``signal(fence, target)``，消费者必须
    ``wait(fence, target)`` 之后才能访问共享资源。只有生产者侧真的
    ``signal`` 到 target 才放行——消费者自己 ``wait`` 了不代表写完，
    这正是原实现放行过早、GPU0 覆盖 GPU1 未完成帧的根因。

    ``spec`` 可以是帧号、围栏值或字典；字典键与显式参数同名。
    """

    opts = _mapping(spec)
    if not opts and isinstance(spec, (int, float)) and not isinstance(spec, bool):
        frame = spec if frame is None else frame

    producer = _as_int(_pick(opts, producer, "producer", "producer_adapter", default=0), 0)
    consumer = _as_int(_pick(opts, consumer, "consumer", "consumer_adapter", default=1), 1)
    frame = _as_int(_pick(opts, frame, "frame", "frame_index", default=0), 0)
    access = str(_pick(opts, access, "access", default="write")).lower()
    resource = _pick(opts, resource, "resource", "texture", "texture_id", default=None)

    target = _pick(opts, fence_value, "fence_value", "value", "signal_value", default=None)
    if target is None:
        target = frame + 1
    target = _as_int(target, frame + 1)
    if target <= 0:
        target = frame + 1

    producer_failed = _as_bool(
        _pick(opts, producer_failed, "producer_failed", "failed", default=False)
    )
    timeout_frames = _pick(opts, timeout_frames, "timeout_frames", default=None)
    blocked_frames = _as_int(
        _pick(opts, blocked_frames, "blocked_frames", default=0), 0
    )

    listed = opts.get("signaled")
    signaled = _pick(opts, signaled, "signaled_value", "producer_value", default=listed)
    listed = opts.get("waited")
    waited = _pick(opts, waited, "waited_value", "consumer_value", default=listed)
    signaled_value = None if signaled is None else _as_int(signaled, 0)
    waited_value = None if waited is None else _as_int(waited, 0)

    is_write = access not in ("read", "read_only", "r", "sr", "none")
    if access in ("none", "no_access"):
        is_write = False

    cross = producer != consumer
    if not cross:
        hazard = "none"
        fence_required = False
        ready = True
        reason = "same_adapter_no_cross_fence"
    else:
        hazard = "cross_write" if is_write else "cross_read"
        # 共享纹理即使只读，也必须等生产者的写入完成，因此一律需要围栏。
        fence_required = access not in ("none", "no_access")
        waited_ok = waited_value is not None and waited_value >= target
        signaled_ok = signaled_value is not None and signaled_value >= target
        waited_beyond_signal = bool(
            waited_ok and not signaled_ok
        )
        if producer_failed:
            ready = False
            reason = "producer_adapter_failed"
        elif signaled_ok:
            # 只有生产者真的 signal 过才证明写入完成；消费者侧的 wait 值
            # 只是一个等待请求，不能反过来证明生产端已完成。
            ready = True
            reason = "wait_already_satisfied" if waited_ok else "fence_signaled"
        else:
            ready = False
            reason = (
                "wait_without_signal" if waited_beyond_signal else "waiting_for_producer"
            )

    blocked = not ready
    timed_out = bool(
        blocked
        and timeout_frames is not None
        and blocked_frames >= max(0, _as_int(timeout_frames, 0))
    )
    deadlock = bool(blocked and producer_failed)

    signal = {
        "adapter": producer,
        "value": target,
        "frame": frame,
        "resource": resource,
    }
    wait = {
        "adapter": consumer,
        "value": target,
        "frame": frame,
        "resource": resource,
    }

    return {
        "ok": ready,
        "ready": ready,
        "blocked": blocked,
        "cross_adapter": cross,
        "same_adapter": not cross,
        "fence_required": fence_required,
        "producer": producer,
        "consumer": consumer,
        "frame": frame,
        "resource": resource,
        "access": access,
        "hazard": hazard,
        "signal": signal,
        "wait": wait,
        "target_value": target,
        "signaled_value": signaled_value,
        "waited_value": waited_value,
        "timed_out": timed_out,
        "deadlock": deadlock,
        "reason": reason,
        # 审计友好的摘要：跨卡写-写是必须阻塞的 hazard。
        "hazardous": bool(cross and is_write and not ready),
    }


# ---------------------------------------------------------------------------
# 2. 命令缓冲按帧分区
# ---------------------------------------------------------------------------


def frame_partition(
    spec: Any = None,
    *args: Any,
    frame: Any = None,
    frames_in_flight: Any = None,
    adapters: Any = None,
    completed: Any = None,
    slot_state: Any = None,
    command_buffers: Any = None,
) -> dict[str, Any]:
    """按 (帧槽位, 适配器) 分配命令缓冲，并判断槽位能否复用。

    原实现只有一条全局命令缓冲环，帧 N 的录制会覆盖仍被 GPU 执行的
    帧 N-K。这里每个适配器各有 ``frames_in_flight`` 条命令缓冲，键为
    ``cb:{adapter}:{slot}``；同一槽位的上一帧未完成时必须等待。

    ``completed`` 可以是已完成帧号的列表；未提供时按“上一帧仍在途”处理，
    这是安全的保守假设。
    """

    opts = _mapping(spec)
    if not opts:
        if isinstance(spec, (int, float)) and not isinstance(spec, bool):
            frame = spec if frame is None else frame
        elif isinstance(spec, (list, tuple)):
            frames = []
            for item in spec:
                one = frame_partition(item, frames_in_flight=frames_in_flight,
                                      adapters=adapters, completed=completed,
                                      slot_state=slot_state)
                if "usable" in one:
                    frames.append(one)
            return {
                "ok": all(item["usable"] for item in frames),
                "frames": frames,
                "count": len(frames),
            }

    frame = _as_int(_pick(opts, frame, "frame", "frame_index", default=0), 0)
    frames_in_flight = _as_int(
        _pick(opts, frames_in_flight, "frames_in_flight", default=DEFAULT_FRAMES_IN_FLIGHT),
        DEFAULT_FRAMES_IN_FLIGHT,
    )
    adapters = _as_int(_pick(opts, adapters, "adapters", "adapter_count",
                             default=DEFAULT_ADAPTERS), DEFAULT_ADAPTERS)
    warnings: list[str] = []
    if frames_in_flight < 1:
        warnings.append("frames_in_flight must be >= 1; clamped")
        frames_in_flight = 1
    if adapters < 1:
        warnings.append("adapters must be >= 1; clamped")
        adapters = 1

    state = _mapping(slot_state)
    completed_values = _pick(opts, completed, "completed", "completed_frames",
                             default=state.get("completed"))
    if isinstance(completed_values, Mapping):
        completed_values = list(completed_values.keys())
    if completed_values is None:
        completed: set[int] | None = None
    elif isinstance(completed_values, (int, float, str)):
        completed = {_as_int(completed_values, -1)}
    else:
        completed = {_as_int(v, -1) for v in completed_values}
        completed.discard(-1)

    buffers = _pick(opts, command_buffers, "command_buffers", default=None)
    existing: dict[str, tuple[int, int] | None] = {}
    alias_pair: list[str] | None = None
    if isinstance(buffers, Mapping):
        for key, value in buffers.items():
            owner: tuple[int, int] | None
            if isinstance(value, Mapping):
                owner = (
                    _as_int(value.get("adapter"), -1),
                    _as_int(value.get("slot"), -1),
                )
            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                owner = (_as_int(value[0], -1), _as_int(value[1], -1))
            else:
                owner = None
            existing[str(key)] = owner
        # 形如 {"cb:0:0": "cb:1:0"} 的共享会造成跨槽位别名。
        seen_targets: dict[str, str] = {}
        for key, owner in existing.items():
            if owner is None:
                continue
            target = f"cb:{owner[0]}:{owner[1]}"
            if owner[0] >= 0 and owner[1] >= 0 and target != key:
                alias_pair = [key, target]
                break
            if target in seen_targets:
                alias_pair = [seen_targets[target], key]
                break
            seen_targets[target] = key
    elif isinstance(buffers, (list, tuple)):
        for item in buffers:
            existing[str(item)] = None

    adapter = frame % adapters
    slot = frame % frames_in_flight
    buffer_key = f"cb:{adapter}:{slot}"
    payload = {
        "frame": frame,
        "adapter": adapter,
        "slot": slot,
        "buffer_key": buffer_key,
        "ring_size": frames_in_flight,
        "adapters": adapters,
    }

    prior_frame = frame - frames_in_flight
    reused = prior_frame >= 0
    if not reused:
        prior_complete = True
    elif completed is None:
        prior_complete = False
    else:
        prior_complete = prior_frame in completed

    if completed is not None:
        # 同一槽位的所有更早帧（slot, slot+fif, ...）都必须完成；否则环形
        # 队列里前面的条目仍在飞行，覆盖它会写出错帧。
        older = [f for f in range(slot, frame, frames_in_flight)
                 if f not in completed]
        if older:
            prior_complete = False

    wait_required = reused and not prior_complete
    usable = not wait_required

    if alias_pair is None and existing:
        owner = existing.get(buffer_key)
        if owner is not None and owner[0] >= 0:
            if owner != (adapter, slot):
                alias_pair = [buffer_key, f"cb:{owner[0]}:{owner[1]}"]
    alias_conflict = alias_pair is not None

    partition_map = [
        {
            "adapter": a,
            "slot": s,
            "buffer_key": f"cb:{a}:{s}",
            "frame_mod": (a, s),
        }
        for a in range(adapters)
        for s in range(frames_in_flight)
    ]

    def _owner_of(key: str) -> tuple[int, int] | None:
        parts = str(key).split(":")
        if len(parts) >= 3:
            try:
                return (int(parts[-2]), int(parts[-1]))
            except ValueError:
                return None
        return None

    alias_cross = False
    if alias_pair:
        left, right = _owner_of(alias_pair[0]), _owner_of(alias_pair[1])
        alias_cross = bool(
            left is not None and right is not None and left[0] != right[0]
        )

    return {
        "ok": usable and not alias_conflict,
        "usable": usable,
        "wait_required": wait_required,
        "reused": reused,
        "prior_frame": prior_frame if reused else None,
        "prior_complete": prior_complete,
        "alias_conflict": alias_conflict,
        "alias_pair": alias_pair,
        "warnings": warnings,
        **payload,
        "total_buffers": adapters * frames_in_flight,
        "partitions": partition_map,
        # 正常分区下两卡同一帧号使用各自的命令缓冲，不存在跨卡别名。
        "cross_adapter_buffer_alias": alias_cross,
        "reason": (
            "buffer_aliased"
            if alias_conflict
            else (
                "slot_available"
                if not reused
                else ("slot_reusable" if prior_complete else "wait_for_prior_frame")
            )
        ),
    }


# ---------------------------------------------------------------------------
# 3. 统一显存预算
# ---------------------------------------------------------------------------


def _normalize_allocation(item: Any, index: int, adapters: int) -> dict[str, Any]:
    data = _mapping(item)
    alloc_id = str(data.get("id", data.get("name", f"alloc-{index}")))
    size = _as_int(
        data.get("bytes", data.get("size", data.get("byte_size", 0))), 0
    )
    shared = _as_bool(
        data.get("shared", data.get("cross_adapter", data.get("cross_gpu", False)))
    )
    used = data.get("adapters", data.get("adapter_set"))
    if used is None:
        if shared:
            used = list(range(adapters))
        else:
            used = [data.get("adapter", 0)]
    elif isinstance(used, (int, str)):
        used = [used]
    used_set = sorted({_as_int(a, 0) for a in used})
    if not used_set:
        used_set = [0]

    copies = data.get("copies")
    if copies is None:
        copies = 1 if shared else len(used_set)
    copies = max(1, _as_int(copies, 1))

    return {
        "id": alloc_id,
        "bytes": max(0, size),
        "shared": shared,
        "adapters": used_set,
        "copies": copies,
    }


def vram_union(
    spec: Any = None,
    *args: Any,
    allocations: Any = None,
    budget: Any = None,
    adapters: Any = None,
    adapter_budgets: Any = None,
) -> dict[str, Any]:
    """合并多适配器显存预算。

    原实现让每个适配器各算一份共享纹理，聚合预算翻倍。这里同一分配 ID
    只计一次；只有 ``shared=False`` 的逐卡副本才乘以副本数。返回值同时
    给出朴素相加（用于暴露原缺陷）与去重后的统一预算。
    """

    opts = _mapping(spec)
    adapters = _as_int(_pick(opts, adapters, "adapters", "adapter_count",
                             default=DEFAULT_ADAPTERS), DEFAULT_ADAPTERS)
    if adapters < 1:
        adapters = 1

    raw = _pick(opts, allocations, "allocations", "resources", default=None)
    if raw is None and isinstance(spec, (list, tuple)):
        raw = list(spec)
    if raw is None and isinstance(spec, (int, float)) and not isinstance(spec, bool):
        raw = [{"id": "alloc-0", "bytes": spec, "shared": True,
                "adapters": list(range(adapters))}]
    if raw is None:
        raw = []
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        # 单个分配的大小：视为一张跨适配器共享分配。
        raw = [{"id": "alloc-0", "bytes": raw, "shared": True,
                "adapters": list(range(adapters))}]
    if isinstance(raw, (str, bytes)):
        raw = [raw]
    if isinstance(raw, Mapping):
        # 形如 {id: bytes} 或 {id: {..}} 的映射。
        expanded = []
        for key, value in raw.items():
            if isinstance(value, Mapping):
                entry = dict(value)
                entry.setdefault("id", key)
            else:
                entry = {"id": key, "bytes": value}
            expanded.append(entry)
        raw = expanded
    if not isinstance(raw, (list, tuple)):
        raw = []

    budget = _pick(opts, budget, "budget", "budget_bytes", default=None)
    budget = None if budget is None else _as_int(budget, 0)

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, item in enumerate(raw):
        alloc = _normalize_allocation(item, index, adapters)
        key = alloc["id"]
        if key not in merged:
            merged[key] = alloc
            order.append(key)
        else:
            existing = merged[key]
            # 同一分配被多个适配器分别登记：合并可见集合，共享标记取或。
            existing["adapters"] = sorted(
                set(existing["adapters"]) | set(alloc["adapters"])
            )
            existing["shared"] = existing["shared"] or alloc["shared"]
            existing["bytes"] = max(existing["bytes"], alloc["bytes"])
            existing["copies"] = max(existing["copies"], alloc["copies"])

    per_adapter = {a: 0 for a in range(adapters)}
    unified = 0
    naive = 0
    entries: list[dict[str, Any]] = []

    for key in order:
        alloc = merged[key]
        size = alloc["bytes"]
        seen = [a for a in alloc["adapters"] if 0 <= a < adapters]
        if not seen:
            seen = [0]
        if alloc["shared"]:
            unified_copies = 1
            naive_copies = len(seen)
            # 朴素统计下每个适配器都把共享分配整份计入自己的显存。
            per_adapter_share = size
        else:
            unified_copies = alloc["copies"]
            naive_copies = alloc["copies"]
            # 私有副本按可见适配器均摊，保证 per-adapter 合计等于总量。
            per_adapter_share = int(
                round(size * alloc["copies"] / len(seen))
            ) if seen else 0
        for adapter in seen:
            per_adapter[adapter] += per_adapter_share
        naive += size * naive_copies
        unified += size * unified_copies
        entries.append(
            {
                **alloc,
                "naive_bytes": size * naive_copies,
                "unified_bytes": size * unified_copies,
                "per_adapter_bytes": per_adapter_share,
                "deduplicated": alloc["shared"] and naive_copies > 1,
            }
        )

    saved = naive - unified
    over = budget is not None and unified > budget
    naive_over = budget is not None and naive > budget

    adapter_budget_map = _mapping(adapter_budgets)
    if adapter_budget_map:
        per_adapter_budget = {
            _as_int(k, 0): _as_int(v, 0) for k, v in adapter_budget_map.items()
        }
    else:
        per_adapter_budget = {a: budget for a in range(adapters)} if budget else {}

    return {
        "ok": not over,
        "adapters": adapters,
        "allocations": entries,
        "unique_allocations": len(entries),
        "naive_bytes": naive,
        "unified_bytes": unified,
        "saved_bytes": saved,
        "double_counted": naive > unified,
        "deduplicated": naive > unified,
        "naive_over_budget": naive_over,
        "over_budget": over,
        "budget": budget,
        "remaining": None if budget is None else max(0, budget - unified),
        "utilization": None
        if not budget
        else round(unified / budget, 6),
        "per_adapter_naive": per_adapter,
        "per_adapter_budget": per_adapter_budget,
        "reason": (
            "shared_allocations_deduplicated"
            if naive > unified
            else "no_shared_duplicates"
        ),
    }


# ---------------------------------------------------------------------------
# 4. 交换链重建与帧号对齐
# ---------------------------------------------------------------------------


def swapchain_rebuild(
    spec: Any = None,
    *args: Any,
    frame: Any = None,
    last_presented: Any = None,
    inflight: Any = None,
    adapters: Any = None,
    frames_in_flight: Any = None,
    reason: Any = None,
    width: Any = None,
    height: Any = None,
    format: Any = None,
    adapter_failed: Any = None,
) -> dict[str, Any]:
    """在帧边界上重建交换链，并保持 AFR 帧号连续。

    重建必须满足两个条件：当前帧严格晚于最后已呈现帧，且所有在途帧
    已经排空。新交换链的起始帧号取 ``last_presented + 1``，因此不会再
    出现“拖屏后帧号归零、两卡奇偶错位”的问题。
    """

    opts = _mapping(spec)
    if not opts and isinstance(spec, (int, float)) and not isinstance(spec, bool):
        frame = spec if frame is None else frame

    frame = _as_int(_pick(opts, frame, "frame", "frame_index", default=0), 0)
    last_presented = _as_int(
        _pick(opts, last_presented, "last_presented", "presented", default=-1), -1
    )
    adapters = _as_int(_pick(opts, adapters, "adapters", default=DEFAULT_ADAPTERS),
                       DEFAULT_ADAPTERS)
    if adapters < 1:
        adapters = 1
    frames_in_flight = _as_int(
        _pick(opts, frames_in_flight, "frames_in_flight",
              default=DEFAULT_FRAMES_IN_FLIGHT),
        DEFAULT_FRAMES_IN_FLIGHT,
    )
    reason = str(_pick(opts, reason, "reason", default="window_moved"))
    width = _as_int(_pick(opts, width, "width", default=1920), 1920)
    height = _as_int(_pick(opts, height, "height", default=1080), 1080)
    format = str(_pick(opts, format, "format", "swapchain_format",
                       default="r8g8b8a8_unorm_srgb"))
    adapter_failed = _pick(opts, adapter_failed, "adapter_failed", default=None)

    raw_inflight = _pick(opts, inflight, "inflight", "in_flight", default=[])
    if isinstance(raw_inflight, Mapping):
        raw_inflight = list(raw_inflight.values())
    inflight_frames = _normalize_frames(raw_inflight)
    pending = [item for item in inflight_frames if not item["complete"]]
    drained = not pending

    prefer_parity = _as_bool(opts.get("prefer_parity", False))

    expected_next = last_presented + 1
    boundary_ok = frame > last_presented
    start_frame = max(frame, expected_next)
    if start_frame <= last_presented:
        start_frame = expected_next
    # 可选：为保持适配器奇偶不变，允许跳过一帧。
    if prefer_parity and (start_frame - expected_next) % 2 == 1:
        start_frame += 1

    requested_owner = frame % adapters
    expected_owner = expected_next % adapters
    aligned_owner = start_frame % adapters
    continuous = start_frame == expected_next
    parity_preserved = aligned_owner == expected_owner
    # 重建后的起始帧本身若归零，就是原来“拖屏后帧号回到 0”的缺陷。
    frame_reset = bool(start_frame == 0 and last_presented >= 0)
    dropped = list(range(expected_next, start_frame)) if start_frame > expected_next else []

    allowed = boundary_ok and drained
    ok = allowed and not frame_reset
    if not boundary_ok:
        reason_text = "not_at_frame_boundary"
    elif not drained:
        reason_text = "inflight_frames_not_drained"
    elif frame_reset:
        reason_text = "frame_index_reset"
    else:
        reason_text = "aligned_to_global_frame"

    return {
        "ok": ok,
        "allowed": allowed,
        "rebuild_frame": start_frame,
        "start_frame": start_frame,
        "base_frame": start_frame,
        "requested_frame": frame,
        "last_presented": last_presented,
        "expected_next_frame": expected_next,
        "boundary_ok": boundary_ok,
        "drained": drained,
        "pending_frames": [item["frame"] for item in pending],
        "drain_required": not drained,
        "dropped_frames": dropped,
        "skipped_frames": dropped,
        "continuous": continuous,
        "frame_index_reset": frame_reset,
        "requested_adapter": requested_owner,
        "expected_adapter": expected_owner,
        "aligned_adapter": aligned_owner,
        "old_adapter": requested_owner,
        "new_adapter": aligned_owner,
        "adapter_parity_preserved": parity_preserved,
        "frames_in_flight": max(1, frames_in_flight),
        "reason": reason_text,
        "trigger": reason,
        "adapter_failed": adapter_failed,
        "fallback_required": adapter_failed is not None,
        "new_swapchain": {
            "width": max(1, width),
            "height": max(1, height),
            "format": format,
            "adapter": aligned_owner,
            "base_frame": start_frame,
            "parity": aligned_owner,
        },
        "requires_wait": not drained,
        "requires_frame_reset": False,
    }


def frame_index_align(
    spec: Any = None,
    *args: Any,
    local_index: Any = None,
    global_index: Any = None,
    rebuild_base: Any = None,
    previous_global: Any = None,
    adapters: Any = None,
    epoch: Any = None,
) -> dict[str, Any]:
    """把重建后的局部帧号映射回全局帧号，并检查连续性/奇偶性。

    交换链重建会让新交换链的局部帧号从 0 开始；AFR 的适配器归属必须
    由全局帧号决定。这里用 ``rebuild_base + local_index`` 恢复全局帧号，
    再与上一帧比较，检测回退、跳号与奇偶错位。
    """

    opts = _mapping(spec)
    if not opts and isinstance(spec, (int, float)) and not isinstance(spec, bool):
        local_index = spec if local_index is None and global_index is None else local_index

    adapters = _as_int(_pick(opts, adapters, "adapters", default=DEFAULT_ADAPTERS),
                       DEFAULT_ADAPTERS)
    if adapters < 1:
        adapters = 1
    epoch = _as_int(_pick(opts, epoch, "epoch", default=0), 0)
    local_index = _as_int(
        _pick(opts, local_index, "local_index", "local", "frame", default=0), 0
    )
    rebuild_base = _pick(opts, rebuild_base, "rebuild_base", "base_frame",
                         default=None)
    global_value = _pick(opts, global_index, "global_index", "global_frame",
                         default=None)

    if global_value is not None:
        global_frame = _as_int(global_value, 0)
        source = "explicit_global"
    elif rebuild_base is not None:
        global_frame = _as_int(rebuild_base, 0) + local_index
        source = "rebuild_base_plus_local"
    else:
        global_frame = local_index + epoch
        source = "local_plus_epoch"

    previous = _pick(opts, previous_global, "previous_global", "previous_frame",
                     default=None)
    previous = None if previous is None else _as_int(previous, -1)
    expected = None if previous is None else previous + 1

    continuous = expected is None or global_frame == expected
    rewind = previous is not None and global_frame <= previous
    duplicate = previous is not None and global_frame == previous
    gap = previous is not None and global_frame > expected

    # 适配器归属只能由全局帧号决定。
    owner = global_frame % adapters
    local_parity = local_index % adapters
    # 重建后的局部帧号与全局帧号的奇偶可能不同，这正是需要重映射的原因；
    # 若直接拿局部帧号选卡就会错位，这里把它作为诊断暴露出来。
    local_parity_misassign = local_parity != owner
    remapped = source == "rebuild_base_plus_local" and local_parity_misassign

    ok = (not rewind) and not duplicate
    if source == "local_plus_epoch":
        # 没有提供重建基址时，无法保证局部→全局映射正确。
        ok = ok and not local_parity_misassign

    return {
        "ok": ok,
        "aligned": ok,
        "global_frame": global_frame,
        "local_index": local_index,
        "rebuild_base": None if rebuild_base is None else _as_int(rebuild_base, 0),
        "offset": None if rebuild_base is None else _as_int(rebuild_base, 0),
        "previous_global": previous,
        "expected_global": expected,
        "source": source,
        "continuous": continuous,
        "rewind": rewind,
        "gap": gap,
        "skipped": 0 if not gap or expected is None else global_frame - expected,
        "duplicate": duplicate,
        "owner_adapter": owner,
        "adapter": owner,
        "local_parity": local_parity,
        "local_parity_misassign": local_parity_misassign,
        "remapped": remapped,
        "parity_aligned": not local_parity_misassign,
        "reason": (
            "global_frame_rewind"
            if rewind
            else (
                "duplicate_frame"
                if duplicate
                else (
                    "local_index_misassigns_adapter"
                    if ok is False
                    else ("remapped_to_global_parity" if remapped else "aligned")
                )
            )
        ),
    }


# ---------------------------------------------------------------------------
# 5. 色彩管线一致性
# ---------------------------------------------------------------------------


def _bit_depth_from_format(fmt: str) -> int | None:
    text = fmt.lower()
    if "r16g16b16a16" in text or "r32" in text:
        return 16
    if "r10g10b10" in text:
        return 10
    if "r8g8b8a8" in text or "b8g8r8a8" in text:
        return 8
    return None


def _normalize_pipeline(value: Any) -> dict[str, Any]:
    data = dict(_mapping(value))
    # 先做旧字段别名归一，再叠加默认值，避免默认值把别名键盖住。
    if "transfer" not in data:
        for alias in ("transfer_function", "encoding", "eotf"):
            if alias in data:
                data["transfer"] = data[alias]
                break
    if "color_space" not in data:
        for alias in ("primaries", "gamut", "colorspace"):
            if alias in data:
                data["color_space"] = data[alias]
                break
    if "shader_gamma" not in data and "manual_gamma" in data:
        data["shader_gamma"] = data["manual_gamma"]
    if "hardware_srgb" not in data:
        for alias in ("srgb_write", "auto_srgb"):
            if alias in data:
                data["hardware_srgb"] = data[alias]
                break
    if "range" not in data and "quantization" in data:
        data["range"] = data["quantization"]

    pipeline = dict(DEFAULT_COLOR_PIPELINE)
    pipeline.update(data)

    fmt = str(pipeline.get("format", DEFAULT_COLOR_PIPELINE["format"]))
    transfer = str(
        pipeline.get("transfer", pipeline.get("transfer_function", "srgb"))
    ).lower()
    if transfer not in TRANSFER_FUNCTIONS:
        transfer = "srgb"
    color_space = str(
        pipeline.get("color_space", pipeline.get("primaries", "srgb"))
    ).lower()
    if color_space not in COLOR_SPACE_RANK:
        color_space = "srgb"
    bit_depth = _as_int(pipeline.get("bit_depth"), 0)
    if bit_depth <= 0:
        bit_depth = _bit_depth_from_format(fmt) or 8
    value_range = str(pipeline.get("range", "full")).lower()
    if value_range in ("limited", "studio", "narrow"):
        value_range = "limited"
    else:
        value_range = "full"

    hardware_srgb = _as_bool(
        pipeline.get(
            "hardware_srgb",
            pipeline.get("srgb_write", pipeline.get("auto_srgb", False)),
        )
    )
    raw_gamma = pipeline.get("shader_gamma", pipeline.get("manual_gamma"))
    shader_gamma: Any
    if raw_gamma is None or raw_gamma is False:
        shader_gamma = None
    elif isinstance(raw_gamma, str):
        text = raw_gamma.strip().lower()
        if text in ("none", "linear", "1", "1.0"):
            shader_gamma = None
        else:
            shader_gamma = text
    else:
        gamma = _as_float(raw_gamma, 1.0)
        shader_gamma = None if abs(gamma - 1.0) < 1e-9 else gamma

    return {
        "transfer": transfer,
        "color_space": color_space,
        "bit_depth": bit_depth,
        "range": value_range,
        "format": fmt,
        "hardware_srgb": hardware_srgb,
        "shader_gamma": shader_gamma,
    }


def _resolved_transfer(pipeline: Mapping[str, Any]) -> tuple[str, bool]:
    """返回 (最终输出传递函数, 是否双重编码)。

    * 交换链本身是 linear，shader 做一次 sRGB 编码：正常，最终为 srgb。
    * 交换链已经是 sRGB（且硬件自动做 _SRGB 写转换），shader 又手动做
      gamma：双重编码，画面偏亮/偏灰。
    """

    base = str(pipeline["transfer"])
    gamma = pipeline["shader_gamma"]

    if gamma is None:
        return base, False
    if base == "linear":
        # 唯一一次编码由 shader 完成。
        return ("srgb" if gamma == "srgb" else "gamma22"), False
    # 交换链已经编码过一次，shader 再编码一次就是双重 gamma。
    return f"double_{base}", True


def color_match(
    spec: Any = None,
    *args: Any,
    reference: Any = None,
    candidate: Any = None,
    target: Any = None,
) -> dict[str, Any]:
    """比较两块适配器的输出管线，判定画面是否会出现色差。

    除逐字段比较外，还会解析“硬件 sRGB + shader 手动 gamma”这种隐式
    双重编码：它会让两卡同一画面明显偏亮/偏灰。返回共同目标管线以及
    需要显式执行的转换步骤。
    """

    opts = _mapping(spec)
    reference_value = _pick(opts, reference, "reference", "a", "gpu0",
                            default=opts.get("pipeline_a"))
    candidate_value = _pick(opts, candidate, "candidate", "b", "gpu1",
                            default=opts.get("pipeline_b"))
    if reference_value is None:
        reference_value = opts if opts and "transfer" in opts else DEFAULT_COLOR_PIPELINE
    if candidate_value is None:
        candidate_value = DEFAULT_COLOR_PIPELINE

    a = _normalize_pipeline(reference_value)
    b = _normalize_pipeline(candidate_value)

    a_output, a_double = _resolved_transfer(a)
    b_output, b_double = _resolved_transfer(b)
    double_gamma = a_double or b_double

    target_data = _mapping(target)
    if target_data:
        common = _normalize_pipeline(target_data)
    else:
        common = {
            "transfer": a_output if a_output == b_output else "srgb",
            "color_space": min(
                (a["color_space"], b["color_space"]),
                key=lambda name: COLOR_SPACE_RANK.get(name, 99),
            ),
            "bit_depth": min(a["bit_depth"], b["bit_depth"]),
            "range": "full" if a["range"] != b["range"] else a["range"],
            "format": a["format"] if a["format"] == b["format"] else None,
        }

    differences: list[str] = []
    for key in ("transfer", "color_space", "bit_depth", "range", "format"):
        if a[key] != b[key]:
            differences.append(key)
    if a_output != b_output:
        differences.append("output_transfer")
    if a["hardware_srgb"] != b["hardware_srgb"]:
        differences.append("hardware_srgb")
    if a["shader_gamma"] != b["shader_gamma"]:
        differences.append("shader_gamma")

    conversions: list[dict[str, Any]] = []
    for adapter_name, pipeline, output in (
        ("reference", a, a_output),
        ("candidate", b, b_output),
    ):
        if pipeline["range"] != common["range"]:
            conversions.append(
                {
                    "adapter": adapter_name,
                    "stage": "shader",
                    "from": pipeline["range"],
                    "to": common["range"],
                    "field": "range",
                }
            )
        if pipeline["color_space"] != common["color_space"]:
            conversions.append(
                {
                    "adapter": adapter_name,
                    "stage": "shader",
                    "from": pipeline["color_space"],
                    "to": common["color_space"],
                    "field": "color_space",
                }
            )
        if output != common["transfer"]:
            conversions.append(
                {
                    "adapter": adapter_name,
                    "stage": "shader",
                    "from": output,
                    "to": common["transfer"],
                    "field": "transfer",
                }
            )

    match = (not differences) and (not double_gamma)
    compatible = not double_gamma
    requires_conversion = bool(conversions)

    return {
        "ok": compatible,
        "match": match,
        "compatible": compatible,
        "identical": match,
        "double_gamma": double_gamma,
        "reference": a,
        "candidate": b,
        "resolved": {"reference": a_output, "candidate": b_output},
        "target": common,
        "differences": differences,
        "conversions": conversions,
        "requires_conversion": requires_conversion,
        "reason": (
            "double_gamma_encoding"
            if double_gamma
            else ("pipelines_identical" if match else "pipeline_normalized")
        ),
    }


# ---------------------------------------------------------------------------
# 6. 各适配器共享纹理
# ---------------------------------------------------------------------------


def shared_texture(
    spec: Any = None,
    *args: Any,
    texture_id: Any = None,
    width: Any = None,
    height: Any = None,
    bytes_per_pixel: Any = None,
    mip_levels: Any = None,
    adapters: Any = None,
    format: Any = None,
    owner: Any = None,
    cross_adapter: Any = None,
) -> dict[str, Any]:
    """登记一张跨适配器共享纹理，并给出围栏/预算约束。

    共享纹理只分配一次，两个适配器拿同一个共享句柄；任何时刻只允许
    一个写者，写入完成后必须 signal 围栏，另一个适配器 wait 后才能读。
    """

    opts = _mapping(spec)
    if not opts and isinstance(spec, (int, float)) and not isinstance(spec, bool):
        texture_id = spec if texture_id is None else texture_id
    texture_id = _pick(opts, texture_id, "texture_id", "id", "name",
                       default="shared-color")
    width = _as_int(_pick(opts, width, "width", default=1920), 1920)
    height = _as_int(_pick(opts, height, "height", default=1080), 1080)
    bytes_per_pixel = _as_int(
        _pick(opts, bytes_per_pixel, "bytes_per_pixel", "bpp", default=4), 4
    )
    mip_levels = _as_int(_pick(opts, mip_levels, "mip_levels", "mips", default=1), 1)
    format = str(_pick(opts, format, "format", default="r8g8b8a8_unorm_srgb"))
    owner = _as_int(_pick(opts, owner, "owner", "producer", default=0), 0)

    raw_adapters = _pick(opts, adapters, "adapters", default=None)
    if raw_adapters is None:
        count = _as_int(opts.get("adapter_count"), DEFAULT_ADAPTERS)
        adapter_list = list(range(max(1, count)))
    elif isinstance(raw_adapters, (int, str)):
        adapter_list = list(range(max(1, _as_int(raw_adapters, 1))))
    else:
        adapter_list = sorted({_as_int(a, 0) for a in raw_adapters})
        if not adapter_list:
            adapter_list = [0]
    cross_adapter = _as_bool(
        _pick(opts, cross_adapter, "cross_adapter", "shared", default=True)
    )

    width = max(1, width)
    height = max(1, height)
    bytes_per_pixel = max(1, bytes_per_pixel)
    mip_levels = max(1, mip_levels)

    base_bytes = _align_up(width * bytes_per_pixel, 256) * height
    # 完整 mip 链约为 4/3；这里按实际层数逐级累加，避免近似误差。
    total_bytes = 0
    for level in range(mip_levels):
        level_w = max(1, width >> level)
        level_h = max(1, height >> level)
        total_bytes += _align_up(level_w * bytes_per_pixel, 256) * level_h

    naive = total_bytes * len(adapter_list)
    unified = total_bytes if cross_adapter else total_bytes * len(adapter_list)
    key = f"tex:{texture_id}"
    handle_source = f"{key}|{format}|{width}x{height}|{mip_levels}"
    handle = "0x" + hashlib.sha1(handle_source.encode("utf-8")).hexdigest()[:16]

    fence_required = bool(cross_adapter and len(adapter_list) > 1)
    return {
        "ok": True,
        "key": key,
        "texture_id": str(texture_id),
        "handle": handle,
        "width": width,
        "height": height,
        "format": format,
        "bytes_per_pixel": bytes_per_pixel,
        "mip_levels": mip_levels,
        "base_level_bytes": base_bytes,
        "bytes": total_bytes,
        "shared": bool(cross_adapter),
        "cross_adapter": bool(cross_adapter),
        "adapters": adapter_list,
        "owner": owner if owner in adapter_list else adapter_list[0],
        "fence_required": fence_required,
        "access_rule": "single_writer_then_fence",
        "naive_bytes": naive,
        "unified_bytes": unified,
        "saved_bytes": naive - unified,
        "instances": len(adapter_list),
        "reason": (
            "shared_allocation_counted_once"
            if cross_adapter
            else "per_adapter_private_copy"
        ),
    }


# ---------------------------------------------------------------------------
# 7. 失败回退与设备丢失
# ---------------------------------------------------------------------------


def device_lost(
    spec: Any = None,
    *args: Any,
    adapter: Any = None,
    frame: Any = None,
    inflight: Any = None,
    generation: Any = None,
    recoverable: Any = None,
    adapters: Any = None,
    reason: Any = None,
) -> dict[str, Any]:
    """处理适配器设备丢失：作废围栏、改派在途帧、转单卡。

    设备丢失后旧围栏可能永远不会被 signal，继续等待就是永久卡死。
    因此提升围栏代际（旧信号全部失效），把在途帧改派给存活适配器，
    并要求调用方重建交换链。
    """

    opts = _mapping(spec)
    if not opts and isinstance(spec, (int, float)) and not isinstance(spec, bool):
        adapter = spec if adapter is None else adapter
    adapter = _as_int(_pick(opts, adapter, "adapter", "failed_adapter",
                            default=1), 1)
    frame = _as_int(_pick(opts, frame, "frame", "frame_index", default=0), 0)
    adapters = _as_int(_pick(opts, adapters, "adapters", default=DEFAULT_ADAPTERS),
                       DEFAULT_ADAPTERS)
    if adapters < 1:
        adapters = 1
    generation = _as_int(_pick(opts, generation, "generation", "fence_generation",
                               default=0), 0)
    reason = str(_pick(opts, reason, "reason", default="device_lost"))

    raw_inflight = _pick(opts, inflight, "inflight", "in_flight", default=[])
    if isinstance(raw_inflight, Mapping):
        raw_inflight = list(raw_inflight.values())
    inflight_frames = _normalize_frames(raw_inflight)
    lost_frames = [
        item["frame"]
        for item in inflight_frames
        if item["adapter"] in (None, adapter) and not item["complete"]
    ]
    if not lost_frames and frame >= 0:
        lost_frames = [frame]

    survivors = [a for a in range(adapters) if a != adapter]
    if recoverable is None:
        recoverable = bool(survivors)
    recoverable = bool(recoverable and survivors)

    new_generation = generation + 1
    reassign_to = survivors[0] if survivors else None
    reassigned = [
        {"frame": frame_index, "from": adapter, "to": reassign_to}
        for frame_index in lost_frames
    ]

    return {
        "ok": recoverable,
        "recoverable": recoverable,
        "adapter": adapter,
        "failed_adapter": adapter,
        "survivors": survivors,
        "mode": "single_adapter" if recoverable else "unavailable",
        "frame": frame,
        "generation": new_generation,
        "fence_generation": new_generation,
        "invalidated_fences": [
            {"adapter": adapter, "up_to": generation}
        ],
        "reassigned": reassigned,
        "lost_frames": lost_frames,
        "rebuild_swapchain": True,
        "fallback": "single_adapter" if recoverable else "none",
        "reason": reason if recoverable else "no_surviving_adapter",
        "next_frame": frame + 1,
    }


def single_fallback(
    spec: Any = None,
    *args: Any,
    failed: Any = None,
    survivors: Any = None,
    adapters: Any = None,
    frame: Any = None,
    inflight: Any = None,
    budget: Any = None,
    generation: Any = None,
    reason: Any = None,
) -> dict[str, Any]:
    """失败后回退到单卡：改派在途帧、收缩统一预算、重建交换链。

    AFR 要求至少两块适配器；只要任一块失败，就必须停用 AFR。预算按
    存活适配器的份额收缩，避免继续沿用双卡聚合预算而超分。
    """

    opts = _mapping(spec)
    if not opts and isinstance(spec, (int, float)) and not isinstance(spec, bool):
        failed = spec if failed is None else failed
    failed = _as_int(_pick(opts, failed, "failed", "failed_adapter", "adapter",
                           default=1), 1)
    adapters = _as_int(_pick(opts, adapters, "adapters", "adapter_count",
                             default=DEFAULT_ADAPTERS), DEFAULT_ADAPTERS)
    if adapters < 1:
        adapters = 1
    frame = _as_int(_pick(opts, frame, "frame", "frame_index", default=0), 0)
    generation = _as_int(_pick(opts, generation, "generation", default=0), 0)
    reason = str(_pick(opts, reason, "reason", default="adapter_failed"))

    raw_survivors = _pick(opts, survivors, "survivors", "alive", default=None)
    if raw_survivors is None:
        survivor_list = [a for a in range(adapters) if a != failed]
    else:
        if isinstance(raw_survivors, (int, str)):
            raw_survivors = [raw_survivors]
        survivor_list = sorted(
            {_as_int(a, -1) for a in raw_survivors if _as_int(a, -1) != failed}
        )
        survivor_list = [a for a in survivor_list if 0 <= a < adapters]

    raw_inflight = _pick(opts, inflight, "inflight", "in_flight", default=[])
    if isinstance(raw_inflight, Mapping):
        raw_inflight = list(raw_inflight.values())
    inflight_frames = _normalize_frames(raw_inflight)
    pending = [
        item["frame"]
        for item in inflight_frames
        if not item["complete"] and item["adapter"] in (None, failed)
    ]

    budget = _pick(opts, budget, "budget", "budget_bytes", default=None)
    budget = None if budget is None else _as_int(budget, 0)
    if budget is None:
        budget_after = None
    else:
        # 双卡聚合预算按存活份额收缩；单卡时保留该卡自身的份额。
        budget_after = int(budget * len(survivor_list) / adapters)

    available = bool(survivor_list)
    survivor = survivor_list[0] if available else None
    reassigned = [
        {"frame": frame_index, "from": failed, "to": survivor}
        for frame_index in pending
    ]

    return {
        "ok": available,
        "available": available,
        "mode": "single_adapter" if available else "unavailable",
        "failed": failed,
        "disabled_adapters": [failed],
        "survivors": survivor_list,
        "survivor": survivor,
        "frame": frame,
        "next_frame": frame + 1,
        "start_frame": frame,
        "reassigned": reassigned,
        "pending_frames": pending,
        "lost_frames": [],
        "generation": generation + 1,
        "fence_generation": generation + 1,
        "swapchain_rebuild": True,
        "rebuild_at_frame": frame + 1,
        "budget_before": budget,
        "budget_after": budget_after,
        "budget": budget_after,
        "afr_enabled": False,
        "reason": reason if available else "no_surviving_adapter",
    }


# ---------------------------------------------------------------------------
# 8. 呈现顺序与统计
# ---------------------------------------------------------------------------


def present_order(
    spec: Any = None,
    *args: Any,
    frames: Any = None,
    adapters: Any = None,
) -> dict[str, Any]:
    """检查并生成按帧号递增的呈现顺序。

    AFR 下两卡独立完成，到达顺序可能与帧号顺序不同。交换链只允许
    单调递增地 present；乱序到达的帧必须等待前一帧，重复帧只能保留
    第一份。
    """

    opts = _mapping(spec)
    adapters = _as_int(_pick(opts, adapters, "adapters", default=DEFAULT_ADAPTERS),
                       DEFAULT_ADAPTERS)
    if adapters < 1:
        adapters = 1

    raw = _pick(opts, frames, "frames", "submissions", default=None)
    if raw is None:
        if isinstance(spec, (list, tuple)):
            raw = list(spec)
        elif isinstance(spec, (int, float)) and not isinstance(spec, bool):
            raw = [spec]
        elif opts:
            raw = opts.get("frames", [])
        else:
            raw = []
    if isinstance(raw, Mapping):
        raw = [raw]

    arrival = _normalize_frames(raw)
    for index, item in enumerate(arrival):
        if item["adapter"] is None:
            item["adapter"] = item["frame"] % adapters
        item["arrival"] = index

    ordered = sorted(arrival, key=lambda item: (item["frame"], item["arrival"]))

    # 到达顺序有没有倒序：后一帧先到，AFR 下这属于正常现象，但必须
    # 由呈现门控挡住，不能按到达顺序 present。
    inversions = 0
    for i in range(len(arrival)):
        for j in range(i + 1, len(arrival)):
            if arrival[i]["frame"] > arrival[j]["frame"]:
                inversions += 1
    arrival_monotonic = inversions == 0

    # 每帧的“被挡住”判定：呈现必须按帧号单调，因此若某帧到达时，
    # 比它号小的帧还没到，这一帧就必须等那个更早的帧。
    waited = False
    max_prior_arrival = -1
    seen: set[int] = set()
    duplicates: list[int] = []
    presents: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for item in ordered:
        frame = item["frame"]
        if frame in seen:
            duplicates.append(frame)
            dropped.append({"frame": frame, "adapter": item["adapter"]})
            continue
        seen.add(frame)
        # max_prior_arrival 是更小帧号中最大的到达序号；它比本帧到得晚，
        # 说明本帧先到、必须等更早的帧呈现。
        waited = max_prior_arrival > item["arrival"]
        max_prior_arrival = max(max_prior_arrival, item["arrival"])
        presents.append(
            {
                "frame": frame,
                "adapter": item["adapter"],
                "arrival": item["arrival"],
                "waited_for_earlier_frame": waited,
                # 只要更早的帧尚未呈现就必须等待；最终按帧号单调 present。
                "wait_for_previous": waited,
            }
        )

    frame_numbers = [item["frame"] for item in presents]
    gaps: list[int] = []
    if frame_numbers:
        full = set(range(min(frame_numbers), max(frame_numbers) + 1))
        gaps = sorted(full - set(frame_numbers))

    wait_frames = [item["frame"] for item in presents if item["wait_for_previous"]]
    # 乱序到达本身不是错误——门控后仍能按帧号正确呈现；重复帧与跳帧才是。
    ok = not duplicates and not gaps
    return {
        "ok": ok,
        "count": len(arrival),
        "presented": len(presents),
        "ordered": frame_numbers,
        "presents": presents,
        "arrival_order": [item["frame"] for item in arrival],
        "arrival_monotonic": arrival_monotonic,
        "requires_reorder": not arrival_monotonic,
        "duplicates": sorted(set(duplicates)),
        "gaps": gaps,
        "inversions": inversions,
        "wait_frames": wait_frames,
        "dropped": dropped,
        "reason": (
            "duplicate_frames"
            if duplicates
            else ("gap" if gaps else ("reordered" if not arrival_monotonic else "in_order"))
        ),
    }


def metrics(
    spec: Any = None,
    *args: Any,
    frame_times: Any = None,
    samples: Any = None,
    adapters: Any = None,
    fence_waits: Any = None,
    budget: Any = None,
    used_bytes: Any = None,
) -> dict[str, Any]:
    """聚合 AFR 观测数据：帧时间、两卡负载均衡、围栏等待与预算。"""

    opts = _mapping(spec)
    adapters = _as_int(_pick(opts, adapters, "adapters", default=DEFAULT_ADAPTERS),
                       DEFAULT_ADAPTERS)
    if adapters < 1:
        adapters = 1

    raw = _pick(opts, frame_times, "frame_times", default=None)
    if raw is None:
        raw = _pick(opts, samples, "samples", "frames", default=None)
    if raw is None:
        if isinstance(spec, (list, tuple)):
            raw = list(spec)
        elif isinstance(spec, (int, float)) and not isinstance(spec, bool):
            raw = [spec]
        else:
            raw = []
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raw = [raw]

    times: list[float] = []
    per_adapter_values: dict[int, list[float]] = {}
    for index, item in enumerate(raw):
        if isinstance(item, Mapping):
            value = item.get("gpu_ms", item.get("ms", item.get("frame_ms",
                                 item.get("time", item.get("frame_time")))))
            if value is None:
                continue
            value = _as_float(value, 0.0)
            adapter = item.get("adapter")
            if adapter is None:
                adapter = index % max(1, adapters)
            adapter = _as_int(adapter, 0)
        else:
            value = _as_float(item, 0.0)
            adapter = index % max(1, adapters)
        if not math.isfinite(value) or value < 0:
            continue
        times.append(value)
        per_adapter_values.setdefault(adapter, []).append(value)

    count = len(times)
    if count:
        mean = sum(times) / count
        variance = sum((value - mean) ** 2 for value in times) / count
        ordered = sorted(times)
        p95_index = min(count - 1, max(0, math.ceil(0.95 * count) - 1))
        p95 = ordered[p95_index]
        stddev = math.sqrt(variance)
        fps = 1000.0 / mean if mean > 0 else 0.0
    else:
        mean = variance = stddev = p95 = fps = 0.0

    per_adapter: dict[str, Any] = {}
    for adapter in range(adapters):
        values = per_adapter_values.get(adapter, [])
        per_adapter[str(adapter)] = {
            "count": len(values),
            "mean_ms": round(sum(values) / len(values), 6) if values else 0.0,
            "max_ms": max(values) if values else 0.0,
        }
    means = [entry["mean_ms"] for entry in per_adapter.values() if entry["count"]]
    imbalance = round(max(means) - min(means), 6) if len(means) > 1 else 0.0

    raw_waits = _pick(opts, fence_waits, "fence_waits", "waits", default=0)
    if isinstance(raw_waits, (list, tuple)):
        wait_total = sum(_as_float(value, 0.0) for value in raw_waits)
        wait_count = len(raw_waits)
    else:
        wait_total = _as_float(raw_waits, 0.0)
        wait_count = _as_int(opts.get("wait_count"), 0)

    budget = _pick(opts, budget, "budget", "budget_bytes", default=None)
    used_bytes = _pick(opts, used_bytes, "used_bytes", "unified_bytes", default=None)
    budget = None if budget is None else _as_int(budget, 0)
    used = None if used_bytes is None else _as_int(used_bytes, 0)
    utilization = None if not budget or used is None else round(used / budget, 6)

    return {
        "ok": count > 0,
        "count": count,
        "mean_ms": round(mean, 6),
        "min_ms": min(times) if times else 0.0,
        "max_ms": max(times) if times else 0.0,
        "variance": round(variance, 6),
        "stddev": round(stddev, 6),
        "p95_ms": p95,
        "fps": round(fps, 3),
        "per_adapter": per_adapter,
        "imbalance_ms": imbalance,
        "balanced": imbalance < 1.0,
        "fence_waits": round(wait_total, 6),
        "fence_wait_count": wait_count,
        "budget": budget,
        "used_bytes": used,
        "utilization": utilization,
        "over_budget": bool(budget is not None and used is not None and used > budget),
        "adapters": adapters,
    }


# ---------------------------------------------------------------------------
# 9. 校验、审计与降级策略
# ---------------------------------------------------------------------------


def validation(spec: Any = None, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """检查 AFR 配置是否满足六项不变量，返回错误与警告列表。"""

    opts = _mapping(spec)
    opts.update(kwargs)
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    adapters = opts.get("adapters", opts.get("adapter_count", DEFAULT_ADAPTERS))
    if isinstance(adapters, (list, tuple)):
        adapter_ids = [_as_int(a, -1) for a in adapters]
        if len(set(adapter_ids)) != len(adapter_ids):
            errors.append("duplicate_adapters")
        adapter_count = len(adapter_ids)
    else:
        adapter_count = _as_int(adapters, DEFAULT_ADAPTERS)
        adapter_ids = list(range(max(1, adapter_count)))
    if adapter_count < 1:
        errors.append("no_adapters")
        adapter_count = 1
    checks["adapters"] = adapter_ids

    failed = opts.get("failed_adapters", opts.get("failed", []))
    if isinstance(failed, (int, float, str)):
        failed = [failed]
    elif not isinstance(failed, (list, tuple, set, frozenset)):
        failed = []
    failed_ids = [_as_int(a, -1) for a in failed] if failed else []
    active = opts.get("active_adapters", adapter_ids)
    if isinstance(active, (int, float, str)):
        active = [active]
    elif not isinstance(active, (list, tuple, set, frozenset)):
        active = adapter_ids
    active_ids = [_as_int(a, -1) for a in active] if active else []
    still_active = sorted(set(failed_ids) & set(active_ids))
    if still_active:
        errors.append("failed_adapter_active")
    checks["failed_adapters"] = failed_ids

    fences = opts.get("fences", opts.get("cross_adapter_fences", []))
    if isinstance(fences, Mapping):
        fences = [fences]
    elif not isinstance(fences, (list, tuple)):
        fences = []
    fence_report = []
    for index, fence in enumerate(fences or []):
        data = _mapping(fence)
        producer = _as_int(data.get("producer", data.get("from", 0)), 0)
        consumer = _as_int(data.get("consumer", data.get("to", 1)), 1)
        required = _as_bool(
            data.get("fence_required", data.get("required", producer != consumer))
        )
        signaled = data.get("signaled", data.get("signaled_value"))
        waited = data.get("waited", data.get("waited_value"))
        frame = _as_int(data.get("frame", 0), 0)
        outcome = cross_adapter_fence(
            producer=producer,
            consumer=consumer,
            frame=frame,
            signaled=signaled,
            waited=waited,
            access=data.get("access", "write"),
            fence_value=data.get("fence_value"),
        )
        fence_report.append(outcome)
        if producer != consumer and required and not outcome["ready"]:
            if not _as_bool(data.get("allow_blocked", False)):
                errors.append(f"missing_cross_adapter_fence:{index}")
    checks["fences"] = fence_report

    buffers = opts.get("command_buffers", opts.get("command_buffer_keys", []))
    if isinstance(buffers, Mapping):
        buffer_list = list(buffers.keys())
    elif isinstance(buffers, (list, tuple)):
        buffer_list = list(buffers)
    elif buffers is None:
        buffer_list = []
    else:
        buffer_list = [buffers]
    if len(buffer_list) != len(set(str(item) for item in buffer_list)):
        errors.append("command_buffer_aliasing")
    checks["command_buffers"] = len(buffer_list)

    if "allocations" in opts:
        unified = vram_union(
            allocations=opts.get("allocations"),
            budget=opts.get("budget", opts.get("budget_bytes")),
            adapters=adapter_count,
        )
        checks["vram"] = unified
        # 共享分配被去重是正确行为，不是错误；只有调用方明确声明按每卡
        # 分别累计（即仍然用朴素相加做预算）时才判为双重统计。
        accounting = str(opts.get("vram_accounting", "unified")).lower()
        if accounting in ("per_adapter", "naive", "separate") and unified["double_counted"]:
            errors.append("vram_double_counted")
        elif _as_bool(opts.get("double_counted", False)):
            errors.append("vram_double_counted")
        if unified["over_budget"]:
            errors.append("vram_over_budget")
    elif "naive_bytes" in opts and "unified_bytes" in opts:
        naive = _as_int(opts.get("naive_bytes"), 0)
        unified_bytes = _as_int(opts.get("unified_bytes"), 0)
        if _as_bool(opts.get("double_counted", False)) or naive > unified_bytes:
            errors.append("vram_double_counted")
        budget = opts.get("budget")
        if budget is not None and unified_bytes > _as_int(budget, 0):
            errors.append("vram_over_budget")
        checks["vram"] = {"naive_bytes": naive, "unified_bytes": unified_bytes}

    swapchain = _mapping(opts.get("swapchain", {}))
    if swapchain:
        result = swapchain_rebuild(swapchain)
        checks["swapchain"] = result
        if not result["boundary_ok"]:
            errors.append("frame_index_rewind")
        if not result["adapter_parity_preserved"]:
            errors.append("adapter_parity_mismatch")
        if "local_index" in swapchain and "global_index" in swapchain:
            aligned = frame_index_align(swapchain)
            checks["frame_align"] = aligned
            if not aligned["ok"]:
                errors.append("frame_index_misaligned")

    color_data = opts.get("color")
    if color_data is not None:
        if isinstance(color_data, Mapping) and (
            "reference" in color_data or "candidate" in color_data
            or "a" in color_data or "b" in color_data
        ):
            result = color_match(color_data)
        elif isinstance(color_data, (list, tuple)) and len(color_data) >= 2:
            result = color_match(reference=color_data[0], candidate=color_data[1])
        else:
            result = color_match(reference=DEFAULT_COLOR_PIPELINE,
                                 candidate=color_data)
        checks["color"] = result
        if result["double_gamma"]:
            errors.append("double_gamma")
        elif not result["compatible"]:
            errors.append("color_pipeline_mismatch")

    present_data = opts.get("present", opts.get("presented"))
    if present_data is not None:
        result = present_order(present_data, adapters=adapter_count)
        checks["present"] = result
        if result["duplicates"]:
            errors.append("duplicate_present")
        if result["gaps"]:
            errors.append("present_frame_gap")
        if (
            result["requires_reorder"]
            and _as_bool(opts.get("present_in_arrival_order", False))
        ):
            # 只有明确按到达顺序呈现时才把乱序当作缺陷；正常路径下
            # 门控会重排，乱序到达属于 AFR 的正常现象。
            errors.append("present_out_of_order")

    if opts.get("frames_in_flight") is not None:
        fif = _as_int(opts.get("frames_in_flight"), DEFAULT_FRAMES_IN_FLIGHT)
        if fif < 1:
            errors.append("invalid_frames_in_flight")
        elif fif > 4:
            warnings.append("large_frames_in_flight")
        checks["frames_in_flight"] = fif

    errors = list(dict.fromkeys(errors))
    warnings = list(dict.fromkeys(warnings))
    return {
        "ok": not errors,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "reason": "ok" if not errors else errors[0],
    }


def audit(spec: Any = None, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """对一次 AFR 运行记录做端到端审计。"""

    record = _mapping(spec)
    record.update(kwargs)
    problems: list[str] = []
    warnings: list[str] = []
    findings: dict[str, Any] = {}

    base = validation(record)
    problems.extend(base["errors"])
    warnings.extend(base["warnings"])
    findings["validation"] = base

    shared = record.get("shared_textures", record.get("shared_texture"))
    shared_reports = []
    if shared is not None:
        if isinstance(shared, Mapping):
            shared = [shared]
        elif not isinstance(shared, (list, tuple)):
            shared = []
        for index, item in enumerate(shared):
            report = shared_texture(item)
            shared_reports.append(report)
            data = _mapping(item)
            fenced = data.get("fenced", data.get("fence_required", None))
            if report["fence_required"] and fenced is False:
                problems.append(f"shared_texture_unfenced:{index}")
        findings["shared_textures"] = shared_reports

    partitions = record.get("partitions")
    if partitions is not None:
        if isinstance(partitions, Mapping):
            frame_report = frame_partition(partitions)
        else:
            frame_report = frame_partition(
                frame=record.get("frame", 0),
                frames_in_flight=record.get("frames_in_flight"),
                adapters=record.get("adapters"),
                completed=record.get("completed"),
                command_buffers=partitions
                if isinstance(partitions, (list, tuple)) else None,
            )
        findings["frame_partition"] = frame_report
        if frame_report["alias_conflict"]:
            problems.append("command_buffer_aliased")
        if frame_report["wait_required"]:
            warnings.append("command_buffer_wait")

    if "swapchain" in record:
        swap = swapchain_rebuild(record.get("swapchain"))
        findings["swapchain"] = swap
        if swap["frame_index_reset"]:
            problems.append("swapchain_frame_reset")
        if not swap["drained"]:
            problems.append("swapchain_rebuild_with_inflight_frames")
        if not swap["adapter_parity_preserved"]:
            problems.append("adapter_parity_mismatch")

    if "color" in record:
        color = color_match(record.get("color"))
        findings["color"] = color
        if color["double_gamma"]:
            problems.append("double_gamma")
        elif not color["match"]:
            warnings.append("color_pipeline_conversion")

    if "device_lost" in record:
        lost_data = record.get("device_lost")
        lost = device_lost(lost_data if lost_data is not None else {})
        findings["device_lost"] = lost
        if not lost["recoverable"]:
            problems.append("unrecoverable_device_lost")

    if "present" in record:
        present = present_order(record.get("present"))
        findings["present"] = present
        if present["duplicates"]:
            problems.append("duplicate_present")
        if present["gaps"]:
            problems.append("present_frame_gap")

    if "allocations" in record:
        vram = vram_union(record.get("allocations"),
                          budget=record.get("budget"),
                          adapters=record.get("adapters"))
        findings["vram"] = vram
        if vram["double_counted"]:
            problems.append("vram_double_counted")

    problems = list(dict.fromkeys(problems))
    warnings = list(dict.fromkeys(warnings))
    fingerprint = determinism(record)["fingerprint"]
    return {
        "ok": not problems,
        "problems": problems,
        "warnings": warnings,
        "findings": findings,
        "fingerprint": fingerprint,
    }


_FALLBACK_ACTIONS = {
    "device_lost": "single_adapter",
    "adapter_failed": "single_adapter",
    "adapter_lost": "single_adapter",
    "fence_timeout": "drain_and_rebuild",
    "fence_stall": "drain_and_rebuild",
    "cross_adapter_fence_timeout": "drain_and_rebuild",
    "swapchain_lost": "drain_and_rebuild",
    "swapchain_failed": "drain_and_rebuild",
    "window_moved": "drain_and_rebuild",
    "command_buffer_conflict": "reduce_frames_in_flight",
    "command_buffer_aliasing": "reduce_frames_in_flight",
    "vram_over_budget": "reduce_frames_in_flight",
    "budget_exceeded": "reduce_frames_in_flight",
    "color_mismatch": "clamp_color_pipeline",
    "double_gamma": "clamp_color_pipeline",
    "unknown": "wait",
}

_FALLBACK_STEPS = {
    "single_adapter": [
        "disable_afr",
        "reassign_inflight_frames",
        "bump_fence_generation",
        "rebuild_swapchain",
        "shrink_budget_to_survivor",
    ],
    "drain_and_rebuild": [
        "stop_submitting_frames",
        "signal_all_queued_fences",
        "drain_inflight_frames",
        "rebuild_swapchain_at_frame_boundary",
        "resume_from_global_frame",
    ],
    "reduce_frames_in_flight": [
        "shrink_command_buffer_ring",
        "wait_for_reused_slots",
        "recalculate_unified_budget",
        "resume_afr",
    ],
    "clamp_color_pipeline": [
        "pick_common_output_pipeline",
        "disable_double_gamma",
        "apply_explicit_shader_conversion",
        "revalidate_color",
    ],
    "wait": ["retry_after_fence_drain"],
}


def fallback(
    spec: Any = None,
    *args: Any,
    kind: Any = None,
    strategies: Any = None,
    reason: Any = None,
) -> dict[str, Any]:
    """根据故障类型选择降级策略，并给出执行步骤。"""

    opts = _mapping(spec)
    if isinstance(spec, str):
        kind = spec if kind is None else kind
    elif isinstance(spec, (int, float)) and not isinstance(spec, bool):
        kind = kind if kind is not None else "adapter_failed"

    kind = str(
        _pick(opts, kind, "kind", "failure", "reason", default="unknown")
    ).lower()
    reason = _pick(opts, reason, "reason", "detail", default=kind)

    raw_strategies = _pick(opts, strategies, "strategies", default=None)
    if raw_strategies is None:
        allowed = list(_FALLBACK_STEPS)
    elif isinstance(raw_strategies, str):
        allowed = [raw_strategies]
    else:
        allowed = [str(item) for item in raw_strategies]

    preferred = _FALLBACK_ACTIONS.get(kind, "wait")
    action = preferred if preferred in allowed else (allowed[0] if allowed else "none")
    available = action in _FALLBACK_STEPS
    steps = list(_FALLBACK_STEPS.get(action, _FALLBACK_STEPS["wait"]))

    return {
        "ok": available,
        "action": action,
        "preferred": preferred,
        "kind": kind,
        "reason": str(reason),
        "steps": steps,
        "enabled": available,
        "allowed": allowed,
        "afr_enabled": action not in ("single_adapter", "none"),
    }


# ---------------------------------------------------------------------------
# 10. 确定性指纹
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    state_method = getattr(value, "state", None)
    if callable(state_method):
        return _canonical(state_method())
    if hasattr(value, "__dict__"):
        public = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        return _canonical(public)
    return repr(value)


def determinism(
    spec: Any = None, *args: Any, repeats: Any = None
) -> dict[str, Any]:
    """状态指纹：与字典插入顺序无关，可重复计算。

    返回 ``fingerprint`` 以及多次计算的 ``runs``，便于断言 AFR 状态机
    在不同到达顺序下得到同一个最终状态。
    """

    payload = _canonical(spec)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    count = max(1, _as_int(repeats, 1))
    runs = [fingerprint for _ in range(count)]
    return {
        "fingerprint": fingerprint,
        "hash": fingerprint,
        "deterministic": len(set(runs)) == 1,
        "runs": runs,
        "canonical": payload,
        "repeats": count,
    }


# ---------------------------------------------------------------------------
# 11. 端到端协调器
# ---------------------------------------------------------------------------


@dataclass
class AFRCoordinator:
    """把六项修复串起来的 AFR 帧协调器。

    典型用法::

        coord = AFRCoordinator(adapters=2, frames_in_flight=2)
        for frame in range(4):
            plan = coord.begin_frame(frame)
            assert plan["ok"]
            coord.complete_frame(frame)
        coord.lose_adapter(1, frame=4)

    协调器只做决策，不触碰真实图形 API；所有状态都可序列化。
    """

    adapters: int = DEFAULT_ADAPTERS
    frames_in_flight: int = DEFAULT_FRAMES_IN_FLIGHT
    budget_bytes: int = DEFAULT_BUDGET_BYTES
    color_pipeline: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_COLOR_PIPELINE)
    )
    # 每块适配器的实际输出管线；缺省视为与基准一致。两卡不一致正是
    # “同一画面颜色略有差异”的来源，begin_frame 会按当前适配器检查。
    adapter_pipelines: dict[int, dict[str, Any]] = field(default_factory=dict)
    shared_allocations: list[dict[str, Any]] = field(default_factory=list)
    completed: set[int] = field(default_factory=set)
    failed_adapters: set[int] = field(default_factory=set)
    fence_generation: int = 0
    fence_value: int = 0
    last_presented: int = -1
    start_frame: int = 0
    afr_enabled: bool = True
    swapchain_epoch: int = 0

    # -- 帧调度 ------------------------------------------------------------

    def active_adapters(self) -> list[int]:
        return [a for a in range(self.adapters) if a not in self.failed_adapters]

    def adapter_for(self, frame: int) -> int:
        active = self.active_adapters()
        if not active:
            return 0
        if self.afr_enabled:
            return active[frame % len(active)]
        return active[0]

    def begin_frame(self, frame: Any = None) -> dict[str, Any]:
        """返回本帧的适配器、命令缓冲、围栏与色彩检查结果。"""

        frame = _as_int(frame, self.start_frame)
        partition = frame_partition(
            frame=frame,
            frames_in_flight=self.frames_in_flight,
            adapters=self.adapters,
            completed=self.completed,
        )
        adapter = self.adapter_for(frame)
        active = self.active_adapters()
        if not active:
            return {
                "ok": False,
                "frame": frame,
                "adapter": None,
                "partition": partition,
                "fallback": "unavailable",
                "reason": "no_surviving_adapter",
            }

        fences: list[dict[str, Any]] = []
        previous_frame = frame - 1
        if self.afr_enabled and previous_frame >= 0:
            producer = self.adapter_for(previous_frame)
            consumer = adapter
            if producer != consumer:
                fences.append(
                    cross_adapter_fence(
                        producer=producer,
                        consumer=consumer,
                        frame=frame,
                        signaled=self.fence_value
                        if previous_frame in self.completed
                        else 0,
                        fence_value=frame,
                    )
                )
        fence_ok = all(item["ready"] for item in fences)

        color = color_match(
            reference=self.color_pipeline,
            candidate=self.adapter_pipelines.get(adapter, self.color_pipeline),
        )

        plan = {
            "ok": partition["usable"] and fence_ok and color["ok"],
            "frame": frame,
            "adapter": adapter,
            "active_adapters": active,
            "afr_enabled": self.afr_enabled,
            "partition": partition,
            "fences": fences,
            "color": color,
            "fallback": None,
            "reason": "ready",
        }
        if not partition["usable"]:
            plan["ok"] = False
            plan["reason"] = "wait_for_command_buffer"
        elif not fence_ok:
            plan["ok"] = False
            plan["reason"] = "wait_for_cross_adapter_fence"
        elif not color["ok"]:
            plan["ok"] = False
            plan["reason"] = "color_pipeline_mismatch"
        return plan

    def complete_frame(self, frame: Any = None, adapter: Any = None) -> dict[str, Any]:
        frame = _as_int(frame, self.start_frame)
        adapter = self.adapter_for(frame) if adapter is None else _as_int(adapter, 0)
        self.completed.add(frame)
        self.fence_value = max(self.fence_value, frame + 1)
        self.last_presented = max(self.last_presented, frame)
        return {
            "ok": True,
            "frame": frame,
            "adapter": adapter,
            "fence_value": self.fence_value,
            "last_presented": self.last_presented,
        }

    def unified_budget(self) -> dict[str, Any]:
        return vram_union(
            allocations=self.shared_allocations,
            budget=self.budget_bytes,
            adapters=self.adapters,
        )

    def rebuild_swapchain(self, frame: Any = None, **kwargs: Any) -> dict[str, Any]:
        # 默认重建点是“下一个尚未呈现的全局帧”，而不是从 0 重来。
        # 调用方显式给出的帧号会原样交给 swapchain_rebuild 校验，这样
        # “把帧号归零重建”仍会被拒绝并暴露出来，而不是被悄悄改掉。
        if frame is None:
            frame = max(self.start_frame, self.last_presented + 1)
        result = swapchain_rebuild(
            frame=_as_int(frame, self.start_frame),
            last_presented=self.last_presented,
            inflight=[
                {"frame": f, "complete": f in self.completed}
                for f in range(
                    self.last_presented + 1 - self.frames_in_flight,
                    self.last_presented + 1,
                )
                if f >= 0
            ],
            adapters=self.adapters,
            frames_in_flight=self.frames_in_flight,
            **{k: v for k, v in kwargs.items()
               if k in ("reason", "width", "height", "format", "adapter_failed")},
        )
        if result["allowed"]:
            self.start_frame = result["start_frame"]
            self.swapchain_epoch += 1
        return result

    def lose_adapter(self, adapter: Any = None, frame: Any = None) -> dict[str, Any]:
        adapter = _as_int(adapter, 1)
        frame = _as_int(frame, self.start_frame)
        lost = device_lost(
            adapter=adapter,
            frame=frame,
            generation=self.fence_generation,
            adapters=self.adapters,
        )
        self.failed_adapters.add(adapter)
        self.fence_generation = lost["fence_generation"]
        plan = single_fallback(
            failed=adapter,
            adapters=self.adapters,
            frame=frame,
            budget=self.budget_bytes,
            generation=self.fence_generation,
            inflight=[
                {"frame": f, "complete": f in self.completed,
                 "adapter": self.adapter_for(f)}
                for f in range(max(0, frame - self.frames_in_flight), frame + 1)
            ],
        )
        self.afr_enabled = False
        self.start_frame = plan["start_frame"]
        return {"device_lost": lost, "fallback": plan, "ok": plan["ok"]}

    def state(self) -> dict[str, Any]:
        return {
            "adapters": self.adapters,
            "frames_in_flight": self.frames_in_flight,
            "budget_bytes": self.budget_bytes,
            "color_pipeline": self.color_pipeline,
            "adapter_pipelines": {
                str(k): v for k, v in sorted(self.adapter_pipelines.items())
            },
            "shared_allocations": self.shared_allocations,
            "completed": sorted(self.completed),
            "failed_adapters": sorted(self.failed_adapters),
            "fence_generation": self.fence_generation,
            "fence_value": self.fence_value,
            "last_presented": self.last_presented,
            "start_frame": self.start_frame,
            "afr_enabled": self.afr_enabled,
            "swapchain_epoch": self.swapchain_epoch,
        }
