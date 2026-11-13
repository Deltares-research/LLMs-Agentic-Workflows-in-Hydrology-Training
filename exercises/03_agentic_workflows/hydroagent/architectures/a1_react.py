"""A1 - Single agent (ReAct): one LLM in a loop with tools. The baseline every other design must beat.

    User -> [LLM <-> tools]* -> answer

The model decides which tool to call next, observes the result, and repeats until it answers.
"""

from __future__ import annotations

from langchain.agents import create_agent

from ..tools import get_tools
from .common import DOMAIN_CONTEXT, ArchInfo

INFO = ArchInfo(
    key="a1",
    name="A1 - Single agent (ReAct)",
    pattern="Dynamic orchestration > Single agent (ReAct)",
    summary="One LLM with tools (list parameters, run scenario, compare, plot) in a reason-act-observe loop.",
    use_when="Most tasks. Start here: one agent + good tools.",
    trade_off="Flexible, but the order of steps is up to the model; no guaranteed checks or approvals.",
    example_query="What happens to peak and low flows at Epinal if soils become 30% less permeable? Plot it.",
)

SYSTEM_PROMPT = DOMAIN_CONTEXT + """

You are a hydrological modelling assistant. Use the tools to answer with model evidence:
1. If unsure which parameter represents the user's change, call list_parameters.
2. Run the scenario(s) with run_scenario (a baseline run is made automatically for comparison).
3. Use compare_runs / plot_hydrographs when they help the user.
4. Answer with the key numbers (% change in mean, peak, high and low flow) and a one-line physical explanation.
Do not run more simulations than needed."""

DEFAULT_TOOLSETS = ("parameters", "simulation", "analysis")


def build(llm, checkpointer=None, toolsets=DEFAULT_TOOLSETS, system_prompt: str = SYSTEM_PROMPT, **_):
    return create_agent(llm, get_tools(*toolsets), system_prompt=system_prompt, checkpointer=checkpointer,
                        name="react_agent")
