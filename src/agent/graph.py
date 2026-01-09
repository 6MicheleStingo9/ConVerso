"""Agent graph for CV-based Q&A with multiple personalities.

This module builds a small state graph (using `langgraph`) that:
- normalizes incoming state,
- asks the LLM for answers using different personalities (e.g. "default", "mysterious"),
- and merges the responses into a session history.

The script can be run interactively: it will load a CV PDF from
`../files/ms_cv.pdf`, extract its text, then repeatedly ask the user for
questions and produce answers for each configured personality.
"""

import os
from typing import Annotated
import operator

from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage

from .prompts import get_system_prompt, USER_PROMPT
from .utils import get_llm_instance, create_logger
from .tools import extract_text_from_pdf, get_message_content


# Graph assembly using langgraph
from langgraph.graph import StateGraph

logger = create_logger("agent")


class AgentState(TypedDict, total=False):
    """TypedDict describing the partial agent state used by the graph.

    Fields are optional (total=False) because the graph's entry node will
    normalize missing keys.
    """

    cv_text: str
    question: str
    # 'history' is built in the final node and is expected to be a list of
    # {'question': str, 'answers': dict} entries.
    history: list
    # 'responses' collects personality-specific answers. Annotated with
    # operator.add so langgraph can merge lists from parallel branches.
    responses: Annotated[list, operator.add]


def prepare_state(state: dict):
    """Normalize incoming state into a valid `AgentState`.

    Ensures `history` and `responses` are always lists, and provides
    default empty strings for optional text fields.

    Args:
        state: Arbitrary dict provided to the graph entry point.

    Returns:
        A dict matching the `AgentState` shape.
    """

    history = state.get("history", [])
    if not isinstance(history, list):
        history = [history]

    responses = state.get("responses", [])
    if not isinstance(responses, list):
        responses = [responses]

    return {
        "cv_text": state.get("cv_text", ""),
        "question": state.get("question", ""),
        "history": history,
        "responses": responses,
    }


def answer_default(state: AgentState) -> dict:
    """Produce an answer using the "default" personality."""

    return answer_personality(state, "default")


def answer_mysterious(state: AgentState) -> dict:
    """Produce an answer using the "mysterious" personality."""

    return answer_personality(state, "mysterious")


def answer_personality(state: AgentState, personality: str) -> dict:
    """Query the LLM for an answer using a given personality.

    The function filters previous history entries to include only those
    where an answer for the requested personality already exists. If such
    entries are present, they are formatted and provided as part of the
    user prompt so the model can be consistent with earlier replies.

    Args:
        state: Current agent state containing `cv_text`, `question` and
            optionally `history`.
        personality: A short string identifying the persona to use.

    Returns:
        A dict with a single key `responses` containing a list with a single
        mapping {personality: answer_text} so it can be merged by the graph.
    """

    llm = get_llm_instance()
    system_msg = SystemMessage(content=get_system_prompt(personality))

    history = state.get("history", [])
    # Keep only history entries that have an answer for this personality
    filtered_history = [h for h in history if personality in h.get("answers", {})]

    if filtered_history:
        formatted_history = "\n".join(
            [
                f"Q: {turn['question']}\nA: {turn['answers'][personality]}"
                for turn in filtered_history
            ]
        )
        # logger.info("History found for personality", personality=personality)
        # logger.debug("Formatted history", history=formatted_history)

    else:
        logger.info("No history for personality", personality=personality)
        formatted_history = "None"

    user_msg = HumanMessage(
        content=USER_PROMPT.format(
            cv_text=state["cv_text"],
            question=state["question"],
            history=formatted_history,
        )
    )

    messages = [system_msg, user_msg]
    answer_msg = llm.invoke(messages)
    answer = get_message_content(answer_msg)

    # Return only the responses field; langgraph will merge this from
    # parallel branches using the Annotated[list, operator.add] rule.
    return {"responses": [{personality: answer}]}


def finalize_session(state: AgentState) -> dict:
    """Final node: merge personality responses into the session history.

    This node collects all partial `responses` from the parallel branches,
    merges them into a single answers dict, and appends a new history
    entry with the current question and the merged answers.
    """

    responses = state.get("responses", [])
    question = state.get("question", "")

    answers = {}
    for resp in responses:
        answers.update(resp)

    new_history_entry = {"question": question, "answers": answers}
    history = state.get("history", []) + [new_history_entry]
    return {"history": history}


def build_agent_graph():
    """Constructs and returns the StateGraph for the agent.

    The graph has the following structure:
    - prepare_state -> (answer_default, answer_mysterious) -> finalize_session
    where the two answer nodes run in parallel and their `responses` lists
    are merged by the Annotated[list, operator.add] rule.
    """

    graph = StateGraph(AgentState)

    graph.add_node("prepare_state", prepare_state)
    graph.add_node("answer_default", answer_default)
    graph.add_node("answer_mysterious", answer_mysterious)
    graph.add_node("finalize_session", finalize_session)

    graph.add_edge("prepare_state", "answer_default")
    graph.add_edge("prepare_state", "answer_mysterious")
    # Parallel nodes converge on the finalize node
    graph.add_edge("answer_default", "finalize_session")
    graph.add_edge("answer_mysterious", "finalize_session")

    graph.set_entry_point("prepare_state")
    graph.set_finish_point("finalize_session")
    return graph


if __name__ == "__main__":
    # Path to the CV PDF (relative to this file)
    cv_path = os.path.join(os.path.dirname(__file__), "../files/ms_cv.pdf")
    cv_path = os.path.abspath(cv_path)

    # Extract the plain text from the CV to use as context for the LLM
    cv_text = extract_text_from_pdf(cv_path)

    # Build and compile the graph once before the interactive loop
    graph = build_agent_graph()
    compiled_graph = graph.compile()

    # Keep session-level history across multiple questions
    history = []

    print("=" * 60)
    print("CV Q&A Agent - Multi-Personality")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break

        current_state = {
            "cv_text": cv_text,
            "question": question,
            "history": history,
            "responses": [],
        }

        # Run the compiled graph with the current state
        result = compiled_graph.invoke(current_state)

        # Update history so subsequent questions see the previous turns
        history = result["history"]

        # Log the final state and responses for debugging / audit
        # logger.info("[FINAL AGENT HISTORY]", state=result["history"], pretty=True)
        # logger.info("[FINAL AGENT RESPONSES]", state=result["responses"], pretty=True)

        # Pretty-print responses grouped by personality
        print("\n" + "-" * 60)
        for resp in result["responses"]:
            for personality, answer in resp.items():
                print(f"\n[{personality.upper()}]:\n{answer}")
        print("-" * 60)
