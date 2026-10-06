from __future__ import annotations

import base64
import json
import mimetypes
import sys
import time

from .geometry import distance2d
from .negotiation import build_vehicle_message, build_vehicle_observations
from .prompting import (
    IMAGE_INPUT_ORDER,
    build_discussion_turn_user_prompt,
    build_negotiation_user_prompt,
    build_priority_fallback_user_prompt,
    build_priority_user_prompt,
    build_system_prompt,
)
from .schema import GeminiConfig, NegotiationResult, TargetSpeedAction, VehicleState


TARGET_SPEED_ACTION_LIST = ["TARGET_0", "TARGET_SLOW", "TARGET_NORMAL", "TARGET_FAST"]
TARGET_SPEED_ACTIONS = set(TARGET_SPEED_ACTION_LIST)
TOKEN_USAGE_TOTAL = {
    "prompt_token_count": 0,
    "candidates_token_count": 0,
    "thoughts_token_count": 0,
    "total_token_count": 0,
    "call_count": 0,
}

QWEN_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


GEMINI_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "target_speed_action": {
            "type": "string",
            "enum": TARGET_SPEED_ACTION_LIST,
        },
        "reason": {"type": "string"},
        "key_risks": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["target_speed_action", "reason", "key_risks"],
}

PRIORITY_SCHEMA = {
    "type": "object",
    "properties": {
        "vehicle_id": {"type": "string"},
        "conflict_detected": {"type": "boolean"},
        "priority_score": {"type": "integer"},
        "priority_reason": {"type": "string"},
        "conflict_summary": {"type": "string"},
        "potential_conflict_vehicle_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "vehicle_id",
        "conflict_detected",
        "priority_score",
        "priority_reason",
        "conflict_summary",
        "potential_conflict_vehicle_ids",
    ],
}

TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "speaker_id": {"type": "string"},
        "agree_with_existing_plan": {"type": "boolean"},
        "proposed_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "vehicle_id": {"type": "string"},
                    "target_speed_action": {
                        "type": "string",
                        "enum": TARGET_SPEED_ACTION_LIST,
                    },
                    "short_reason": {"type": "string"},
                },
                "required": ["vehicle_id", "target_speed_action", "short_reason"],
            },
        },
        "message": {"type": "string"},
    },
    "required": [
        "speaker_id",
        "agree_with_existing_plan",
        "proposed_actions",
        "message",
    ],
}

FALLBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "final_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "vehicle_id": {"type": "string"},
                    "target_speed_action": {
                        "type": "string",
                        "enum": TARGET_SPEED_ACTION_LIST,
                    },
                    "reason": {"type": "string"},
                    "key_risks": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["vehicle_id", "target_speed_action", "reason", "key_risks"],
            },
        },
        "fallback_summary": {"type": "string"},
    },
    "required": ["final_actions", "fallback_summary"],
}


def _parse_decision(response_text: str) -> tuple[TargetSpeedAction, str, list[str]]:
    payload = json.loads(response_text)
    target_speed_action = TargetSpeedAction(payload["target_speed_action"])
    reason = str(payload["reason"])
    key_risks = [str(item) for item in payload.get("key_risks", [])]
    return target_speed_action, reason, key_risks


def _guess_mime_type(image_path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    return mime_type or "image/png"


def get_token_usage_total() -> dict:
    return dict(TOKEN_USAGE_TOTAL)


def _token_usage_delta(before: dict, after: dict) -> dict:
    keys = set(before.keys()) | set(after.keys())
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}


def _usage_value(usage, name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(name, 0) or 0)
    return int(getattr(usage, name, 0) or 0)


def _record_token_usage(response) -> None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        usage = getattr(response, "usage", None)
    prompt_tokens = _usage_value(usage, "prompt_token_count")
    if prompt_tokens <= 0:
        prompt_tokens = _usage_value(usage, "prompt_tokens")
    candidate_tokens = _usage_value(usage, "candidates_token_count")
    if candidate_tokens <= 0:
        candidate_tokens = _usage_value(usage, "completion_tokens")
    thoughts_tokens = _usage_value(usage, "thoughts_token_count")
    total_tokens = _usage_value(usage, "total_token_count")
    if total_tokens <= 0:
        total_tokens = _usage_value(usage, "total_tokens")
    if total_tokens <= 0:
        total_tokens = prompt_tokens + candidate_tokens + thoughts_tokens

    TOKEN_USAGE_TOTAL["prompt_token_count"] += prompt_tokens
    TOKEN_USAGE_TOTAL["candidates_token_count"] += candidate_tokens
    TOKEN_USAGE_TOTAL["thoughts_token_count"] += thoughts_tokens
    TOKEN_USAGE_TOTAL["total_token_count"] += total_tokens
    TOKEN_USAGE_TOTAL["call_count"] += 1


def _image_to_data_url(image_path) -> str:
    mime_type = _guess_mime_type(image_path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return "data:%s;base64,%s" % (mime_type, encoded)


def _qwen_messages(system_prompt: str, user_prompt: str, image_paths: list | None) -> list[dict]:
    user_content: list[dict] = [{"type": "text", "text": user_prompt}]
    for image_path in image_paths or []:
        if image_path and image_path.exists():
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_url(image_path)},
                }
            )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _extract_response_text(response) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("LLM response did not contain choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
            elif hasattr(part, "text"):
                text_parts.append(str(part.text))
        return "".join(text_parts).strip()
    raise ValueError("LLM response did not contain text content")


def _validate_json_schema_subset(payload, schema: dict, path: str = "response") -> None:
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(payload, dict):
            raise ValueError("%s must be an object" % path)
        for key in schema.get("required", []):
            if key not in payload:
                raise ValueError("%s missing required key: %s" % (path, key))
        properties = schema.get("properties", {})
        for key, child_schema in properties.items():
            if key in payload:
                _validate_json_schema_subset(payload[key], child_schema, "%s.%s" % (path, key))
    elif schema_type == "array":
        if not isinstance(payload, list):
            raise ValueError("%s must be an array" % path)
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(payload):
                _validate_json_schema_subset(item, item_schema, "%s[%d]" % (path, index))
    elif schema_type == "string":
        if not isinstance(payload, str):
            raise ValueError("%s must be a string" % path)
    elif schema_type == "boolean":
        if not isinstance(payload, bool):
            raise ValueError("%s must be a boolean" % path)
    elif schema_type == "integer":
        if not isinstance(payload, int) or isinstance(payload, bool):
            raise ValueError("%s must be an integer" % path)

    if "enum" in schema and payload not in schema["enum"]:
        raise ValueError("%s must be one of %s" % (path, schema["enum"]))


def _is_non_retryable_auth_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {401, 403}:
        return True
    text = str(exc).lower()
    return "invalid_api_key" in text or "incorrect api key" in text


def _debug_response_excerpt(response_text: str, limit: int = 500) -> str:
    compact = " ".join(str(response_text).split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _call_qwen_json(
    client,
    *,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict,
    image_paths: list | None = None,
    retry_wait_sec: float = 10.0,
    retry_attempts: int = 0,
    extra_body: dict | None = None,
    debug: bool = False,
) -> dict:
    request_extra_body = dict(extra_body or {})
    request_extra_body.setdefault("enable_thinking", False)
    messages = _qwen_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=image_paths,
    )
    attempt = 1
    while True:
        response_text = ""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                extra_body=request_extra_body,
            )
            _record_token_usage(response)
            response_text = _extract_response_text(response)
            payload = json.loads(response_text)
            _validate_json_schema_subset(payload, response_schema)
            return payload
        except Exception as exc:
            if _is_non_retryable_auth_error(exc):
                raise
            if retry_attempts > 0 and attempt >= retry_attempts:
                raise
            if debug:
                detail = ""
                if response_text:
                    detail = " response_excerpt=%s" % _debug_response_excerpt(response_text)
                print(
                    "[cov2v] qwen call failed on attempt %d: %s;%s retrying in %.1fs"
                    % (attempt, exc, detail, retry_wait_sec),
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(retry_wait_sec)
            attempt += 1


def _call_gemini_json(
    client,
    *,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict,
    image_paths: list | None = None,
    retry_wait_sec: float = 10.0,
    retry_attempts: int = 0,
) -> dict:
    from google.genai import types

    contents: list[object] = [user_prompt]
    for image_path in image_paths or []:
        if image_path and image_path.exists():
            contents.append(
                types.Part.from_bytes(
                    data=image_path.read_bytes(),
                    mime_type=_guess_mime_type(image_path),
                )
            )

    # Gemini 2.x turns thinking off with thinking_budget=0; the 3.x models
    # reject that argument (400 INVALID_ARGUMENT) and take thinking_level.
    thinking_config = (
        types.ThinkingConfig(thinking_level="low")
        if model.startswith("gemini-3")
        else types.ThinkingConfig(thinking_budget=0)
    )

    generation_config = types.GenerateContentConfig(
        thinking_config=thinking_config,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=response_schema,
        system_instruction=system_prompt,
    )
    attempt = 1
    while True:
        try:
            response = client.models.generate_content(
                model=model,
                config=generation_config,
                contents=contents,
            )
            _record_token_usage(response)
            return json.loads(response.text)
        except Exception as exc:
            if retry_attempts > 0 and attempt >= retry_attempts:
                raise
            print(
                "[cov2v] gemini call failed on attempt %d: %s; retrying in %.1fs"
                % (attempt, exc, retry_wait_sec),
                file=sys.stderr,
                flush=True,
            )
            time.sleep(retry_wait_sec)
            attempt += 1


def _build_llm_client(gemini_config: GeminiConfig):
    if gemini_config.provider == "qwen":
        from openai import OpenAI

        return OpenAI(
            api_key=gemini_config.api_key,
            base_url=gemini_config.base_url or QWEN_DEFAULT_BASE_URL,
        )

    from google import genai

    return genai.Client(api_key=gemini_config.api_key)


def _call_llm_json(
    client,
    *,
    gemini_config: GeminiConfig,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict,
    image_paths: list | None = None,
) -> dict:
    if gemini_config.provider == "qwen":
        return _call_qwen_json(
            client,
            model=gemini_config.model,
            temperature=gemini_config.temperature,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_schema,
            image_paths=image_paths,
            retry_wait_sec=gemini_config.retry_wait_sec,
            retry_attempts=gemini_config.retry_attempts,
            extra_body=gemini_config.extra_body,
            debug=gemini_config.debug,
        )
    return _call_gemini_json(
        client,
        model=gemini_config.model,
        temperature=gemini_config.temperature,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=response_schema,
        image_paths=image_paths,
        retry_wait_sec=gemini_config.retry_wait_sec,
        retry_attempts=gemini_config.retry_attempts,
    )


def negotiate_with_gemini(
    ego: VehicleState,
    all_states: list[VehicleState],
    *,
    gemini_config: GeminiConfig,
) -> NegotiationResult:
    outbound_message = build_vehicle_message(ego)
    scoped_neighbors = (
        _states_in_range(ego, all_states, gemini_config.cooperative_range_meters)
        if gemini_config.use_other_vehicle_observations
        else []
    )
    observations = (
        build_vehicle_observations(ego, [ego, *scoped_neighbors])
        if gemini_config.use_other_vehicle_observations
        else []
    )

    client = _build_llm_client(gemini_config)
    system_prompt = build_system_prompt(gemini_config.system_prompt_path)
    user_prompt = build_negotiation_user_prompt(
        ego_state=ego,
        cooperative_observations=observations,
        include_strategy_knowledge=gemini_config.use_strategy_knowledge,
    )
    response_payload = _call_llm_json(
        client,
        gemini_config=gemini_config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=GEMINI_DECISION_SCHEMA,
        image_paths=[ego.image_paths.get(name) for name in IMAGE_INPUT_ORDER],
    )
    target_speed_action, reason, key_risks = _parse_decision(json.dumps(response_payload, ensure_ascii=False))

    return NegotiationResult(
        ego_id=ego.vehicle_id,
        frame=ego.frame,
        action=target_speed_action,
        reason=reason,
        key_risks=key_risks,
        cooperative_observations=observations,
        outbound_message=outbound_message,
    )


def _normalize_actions(action_items: list[dict]) -> tuple[tuple[str, str], ...]:
    normalized = []
    for item in action_items:
        vehicle_id = str(item.get("vehicle_id", "")).strip()
        target_speed_action = str(item.get("target_speed_action", "")).strip()
        if vehicle_id and target_speed_action in TARGET_SPEED_ACTIONS:
            normalized.append((vehicle_id, target_speed_action))
    return tuple(sorted(normalized))


def _latest_proposed_action_for_vehicle(
    vehicle_id: str,
    latest_turns: dict[str, dict],
) -> tuple[str, str]:
    own_turn = latest_turns.get(vehicle_id, {})
    for item in own_turn.get("proposed_actions", []):
        if str(item.get("vehicle_id", "")).strip() == vehicle_id:
            action = str(item.get("target_speed_action", "")).strip()
            reason = str(item.get("short_reason", "")).strip()
            if action in TARGET_SPEED_ACTIONS:
                return action, reason or "Recovered from this vehicle's latest discussion proposal."

    for turn in reversed(list(latest_turns.values())):
        for item in turn.get("proposed_actions", []):
            if str(item.get("vehicle_id", "")).strip() == vehicle_id:
                action = str(item.get("target_speed_action", "")).strip()
                reason = str(item.get("short_reason", "")).strip()
                if action in TARGET_SPEED_ACTIONS:
                    return action, reason or "Recovered from the latest discussion proposal."

    return "TARGET_NORMAL", "No valid final target_speed_action was returned for this vehicle; using TARGET_NORMAL as a safe non-punitive fallback."


def _sanitize_final_actions(
    final_payload: dict,
    vehicle_ids: list[str],
    latest_turns: dict[str, dict],
) -> dict:
    expected_ids = set(vehicle_ids)
    sanitized_by_id: dict[str, dict] = {}
    for item in final_payload.get("final_actions", []):
        vehicle_id = str(item.get("vehicle_id", "")).strip()
        target_speed_action = str(item.get("target_speed_action", "")).strip()
        if vehicle_id not in expected_ids or target_speed_action not in TARGET_SPEED_ACTIONS:
            continue
        sanitized_by_id[vehicle_id] = {
            "vehicle_id": vehicle_id,
            "target_speed_action": target_speed_action,
            "reason": str(item.get("reason", "")).strip() or str(item.get("short_reason", "")).strip(),
            "key_risks": [str(risk) for risk in item.get("key_risks", [])],
        }

    for vehicle_id in vehicle_ids:
        if vehicle_id in sanitized_by_id:
            continue
        action, reason = _latest_proposed_action_for_vehicle(vehicle_id, latest_turns)
        sanitized_by_id[vehicle_id] = {
            "vehicle_id": vehicle_id,
            "target_speed_action": action,
            "reason": reason,
            "key_risks": ["Final decision omitted this vehicle; action recovered from discussion history."],
        }

    final_payload["final_actions"] = [sanitized_by_id[vehicle_id] for vehicle_id in vehicle_ids]
    return final_payload


def _states_in_range(
    ego: VehicleState,
    all_states: list[VehicleState],
    cooperative_range_meters: float,
) -> list[VehicleState]:
    ego_xy = (ego.pose.x, ego.pose.y)
    return [
        other
        for other in all_states
        if other.vehicle_id != ego.vehicle_id
        and distance2d(ego_xy, (other.pose.x, other.pose.y)) <= cooperative_range_meters
    ]


def _normalize_conflict_ids(
    payload: dict,
    *,
    ego_id: str,
    allowed_ids: set[str],
) -> dict:
    raw_ids = payload.get("potential_conflict_vehicle_ids", [])
    normalized_ids: list[str] = []
    for item in raw_ids if isinstance(raw_ids, list) else []:
        vehicle_id = str(item)
        if vehicle_id != ego_id and vehicle_id in allowed_ids and vehicle_id not in normalized_ids:
            normalized_ids.append(vehicle_id)
    payload["potential_conflict_vehicle_ids"] = normalized_ids
    return payload


def _build_conflict_groups(priorities: list[dict], vehicle_ids: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    adjacency: dict[str, set[str]] = {vehicle_id: set() for vehicle_id in vehicle_ids}
    conflict_edges: set[tuple[str, str]] = set()
    valid_ids = set(vehicle_ids)
    for item in priorities:
        vehicle_id = str(item["vehicle_id"])
        if vehicle_id not in valid_ids:
            continue
        for other_id in item.get("potential_conflict_vehicle_ids", []):
            if other_id not in valid_ids or other_id == vehicle_id:
                continue
            adjacency[vehicle_id].add(other_id)
            adjacency[other_id].add(vehicle_id)
            conflict_edges.add(tuple(sorted((vehicle_id, other_id))))

    groups: list[list[str]] = []
    visited: set[str] = set()
    for vehicle_id in vehicle_ids:
        if vehicle_id in visited or not adjacency[vehicle_id]:
            continue
        stack = [vehicle_id]
        component: list[str] = []
        visited.add(vehicle_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        groups.append(sorted(component))
    return groups, [list(edge) for edge in sorted(conflict_edges)]


def _actions_agree(latest_turns: dict[str, dict], vehicle_ids: list[str]) -> bool:
    if len(latest_turns) != len(vehicle_ids):
        return False
    normalized = []
    expected_ids = set(vehicle_ids)
    for vehicle_id in vehicle_ids:
        turn = latest_turns.get(vehicle_id)
        if not turn:
            return False
        actions = _normalize_actions(turn["proposed_actions"])
        action_ids = {item[0] for item in actions}
        if action_ids != expected_ids:
            return False
        normalized.append(actions)
    return len(set(normalized)) == 1


def _final_decision_from_agreed_actions(latest_turns: dict[str, dict], vehicle_ids: list[str]) -> dict:
    first_turn = latest_turns[vehicle_ids[0]]
    final_actions = []
    reason_by_vehicle = {
        str(item.get("vehicle_id", "")): str(item.get("short_reason", "Agreed in discussion."))
        for item in first_turn.get("proposed_actions", [])
    }
    for vehicle_id, target_speed_action in _normalize_actions(first_turn.get("proposed_actions", [])):
        final_actions.append(
            {
                "vehicle_id": vehicle_id,
                "target_speed_action": target_speed_action,
                "reason": reason_by_vehicle.get(vehicle_id, "Agreed in discussion."),
                "key_risks": [],
            }
        )
    return {
        "final_actions": final_actions,
        "fallback_summary": "All discussion turns proposed the same final actions.",
    }


def _run_group_discussion(
    client,
    *,
    group_states: list[VehicleState],
    observations_by_vehicle: dict[str, list[dict]],
    priorities: list[dict],
    outbound_by_vehicle: dict[str, object],
    gemini_config: GeminiConfig,
    system_prompt: str,
) -> tuple[list[NegotiationResult], dict]:
    priorities = sorted(
        priorities,
        key=lambda item: (
            0 if item.get("conflict_detected", False) else 1,
            -int(item.get("priority_score", 0)),
            str(item.get("vehicle_id", "")),
        ),
    )
    speaking_order = [str(item["vehicle_id"]) for item in priorities]
    state_by_id = {state.vehicle_id: state for state in group_states}
    priority_by_id = {str(item["vehicle_id"]): item for item in priorities}
    vehicle_ids = [state.vehicle_id for state in group_states]

    transcript: list[dict] = []
    latest_turns: dict[str, dict] = {}
    for round_index in range(1, gemini_config.discussion_rounds + 1):
        for speaker_id in speaking_order:
            state = state_by_id[speaker_id]
            turn_prompt = build_discussion_turn_user_prompt(
                ego_state=state,
                cooperative_observations=observations_by_vehicle[speaker_id],
                own_priority=priority_by_id[speaker_id],
                all_priorities=priorities,
                speaking_order=speaking_order,
                transcript=transcript,
                round_index=round_index,
                include_strategy_knowledge=gemini_config.use_strategy_knowledge,
            )
            turn_payload = _call_llm_json(
                client,
                gemini_config=gemini_config,
                system_prompt=system_prompt,
                user_prompt=turn_prompt,
                response_schema=TURN_SCHEMA,
                image_paths=[state.image_paths.get(name) for name in IMAGE_INPUT_ORDER],
            )
            turn_payload["round_index"] = round_index
            transcript.append(turn_payload)
            latest_turns[speaker_id] = turn_payload
        if _actions_agree(latest_turns, vehicle_ids):
            break

    if _actions_agree(latest_turns, vehicle_ids):
        final_payload = _final_decision_from_agreed_actions(latest_turns, vehicle_ids)
        final_decision_source = "discussion_actions"
    else:
        fallback_speaker_id = speaking_order[0]
        fallback_state = state_by_id[fallback_speaker_id]
        fallback_prompt = build_priority_fallback_user_prompt(
            ego_state=fallback_state,
            cooperative_observations=observations_by_vehicle[fallback_speaker_id],
            own_priority=priority_by_id[fallback_speaker_id],
            all_priorities=priorities,
            speaking_order=speaking_order,
            transcript=transcript,
            include_strategy_knowledge=gemini_config.use_strategy_knowledge,
        )
        final_payload = _call_llm_json(
            client,
            gemini_config=gemini_config,
            system_prompt=system_prompt,
            user_prompt=fallback_prompt,
            response_schema=FALLBACK_SCHEMA,
            image_paths=[fallback_state.image_paths.get(name) for name in IMAGE_INPUT_ORDER],
        )
        final_decision_source = "priority_fallback"
    final_payload = _sanitize_final_actions(final_payload, vehicle_ids, latest_turns)
    final_actions = {
        str(item["vehicle_id"]): item
        for item in final_payload.get("final_actions", [])
    }

    results: list[NegotiationResult] = []
    for state in group_states:
        final_item = final_actions.get(state.vehicle_id, {})
        results.append(
            NegotiationResult(
                ego_id=state.vehicle_id,
                frame=state.frame,
                action=TargetSpeedAction(final_item.get("target_speed_action", "TARGET_NORMAL")),
                reason=str(final_item.get("reason", final_payload.get("fallback_summary", ""))),
                key_risks=[str(item) for item in final_item.get("key_risks", [])],
                cooperative_observations=observations_by_vehicle[state.vehicle_id],
                outbound_message=outbound_by_vehicle[state.vehicle_id],
            )
        )
    group_process = {
        "vehicle_ids": vehicle_ids,
        "priorities": priorities,
        "speaking_order": speaking_order,
        "transcript": transcript,
        "final_decision": final_payload,
        "final_decision_source": final_decision_source,
    }
    return results, group_process


def negotiate_with_gemini_discussion(
    all_states: list[VehicleState],
    *,
    gemini_config: GeminiConfig,
) -> tuple[list[NegotiationResult], dict]:
    client = _build_llm_client(gemini_config)
    system_prompt = build_system_prompt(gemini_config.system_prompt_path)

    if gemini_config.use_other_vehicle_observations:
        scoped_neighbors_by_vehicle = {
            state.vehicle_id: _states_in_range(state, all_states, gemini_config.cooperative_range_meters)
            for state in all_states
        }
        observations_by_vehicle = {
            state.vehicle_id: build_vehicle_observations(state, [state, *scoped_neighbors_by_vehicle[state.vehicle_id]])
            for state in all_states
        }
    else:
        scoped_neighbors_by_vehicle = {state.vehicle_id: [] for state in all_states}
        observations_by_vehicle = {state.vehicle_id: [] for state in all_states}
    outbound_by_vehicle = {
        state.vehicle_id: build_vehicle_message(state)
        for state in all_states
    }

    priorities: list[dict] = []
    allowed_ids_by_vehicle = {
        state.vehicle_id: {other.vehicle_id for other in scoped_neighbors_by_vehicle[state.vehicle_id]}
        for state in all_states
    }
    for state in all_states:
        priority_prompt = build_priority_user_prompt(
            ego_state=state,
            cooperative_observations=observations_by_vehicle[state.vehicle_id],
            include_strategy_knowledge=gemini_config.use_strategy_knowledge,
        )
        priority_payload = _call_llm_json(
            client,
            gemini_config=gemini_config,
            system_prompt=system_prompt,
            user_prompt=priority_prompt,
            response_schema=PRIORITY_SCHEMA,
            image_paths=[state.image_paths.get(name) for name in IMAGE_INPUT_ORDER],
        )
        priority_payload = _normalize_conflict_ids(
            priority_payload,
            ego_id=state.vehicle_id,
            allowed_ids=allowed_ids_by_vehicle[state.vehicle_id],
        )
        priorities.append(priority_payload)

    vehicle_ids = [state.vehicle_id for state in all_states]
    priority_by_id = {str(item["vehicle_id"]): item for item in priorities}
    state_by_id = {state.vehicle_id: state for state in all_states}
    conflict_groups, conflict_edges = _build_conflict_groups(priorities, vehicle_ids)

    grouped_vehicle_ids = {vehicle_id for group in conflict_groups for vehicle_id in group}
    results_by_id: dict[str, NegotiationResult] = {}
    group_processes: list[dict] = []

    for group_index, group_vehicle_ids in enumerate(conflict_groups, start=1):
        group_states = [state_by_id[vehicle_id] for vehicle_id in group_vehicle_ids]
        group_observations = {
            vehicle_id: [
                obs for obs in observations_by_vehicle[vehicle_id]
                if obs.other_id in set(group_vehicle_ids)
            ]
            for vehicle_id in group_vehicle_ids
        }
        group_priorities = [priority_by_id[vehicle_id] for vehicle_id in group_vehicle_ids]
        group_results, group_process = _run_group_discussion(
            client,
            group_states=group_states,
            observations_by_vehicle=group_observations,
            priorities=group_priorities,
            outbound_by_vehicle=outbound_by_vehicle,
            gemini_config=gemini_config,
            system_prompt=system_prompt,
        )
        for result in group_results:
            results_by_id[result.ego_id] = result
        group_process["group_index"] = group_index
        group_processes.append(group_process)

    for state in all_states:
        if state.vehicle_id in grouped_vehicle_ids:
            continue
        result = negotiate_with_gemini(
            state,
            [state, *scoped_neighbors_by_vehicle[state.vehicle_id]],
            gemini_config=gemini_config,
        )
        results_by_id[state.vehicle_id] = result

    results = [results_by_id[state.vehicle_id] for state in all_states]
    discussion_process = {
        "cooperative_range_meters": gemini_config.cooperative_range_meters,
        "use_strategy_knowledge": gemini_config.use_strategy_knowledge,
        "use_other_vehicle_observations": gemini_config.use_other_vehicle_observations,
        "priorities": priorities,
        "conflict_edges": conflict_edges,
        "conflict_groups": conflict_groups,
        "group_discussions": group_processes,
        "non_grouped_vehicle_ids": sorted(set(vehicle_ids) - grouped_vehicle_ids),
    }
    return results, discussion_process
