import pytest
import graphics.mgpu_afr_fence as mod

CASES = [
    ("cross_adapter_fence", (1)),
    ("frame_partition", (1)),
    ("vram_union", (1)),
    ("swapchain_rebuild", (1)),
    ("frame_index_align", (1)),
    ("color_match", (1)),
    ("single_fallback", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("shared_texture", (1)),
    ("metrics", (1)),
    ("present_order", (1)),
    ("device_lost", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
