import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.router import DEFAULT_VERIFICATION_PROMPT, parse_router_output  # noqa: E402


def test_parse_router_output_json_payload():
    raw = (
        '{"description": "people entering the shop",'
        ' "visual_tags": ["red clothes"],'
        ' "needed_facts": ["start_s", "end_s"],'
        ' "constraints": {"limit": 1},'
        ' "verification_prompt": ""}'
    )
    plan, think = parse_router_output(raw)
    assert plan.description == "people entering the shop"
    assert plan.visual_tags == ["red clothes"]
    assert plan.needed_facts == ["start_s", "end_s"]
    assert plan.constraints["limit"] == 1
    assert plan.verification_prompt == DEFAULT_VERIFICATION_PROMPT
    assert think == ""


def test_parse_router_output_fallback_to_text():
    plan, think = parse_router_output("找穿紫色衣服的人")
    assert plan.description == "找穿紫色衣服的人"
    assert plan.visual_tags == []
    assert plan.verification_prompt == DEFAULT_VERIFICATION_PROMPT
    assert think == ""
