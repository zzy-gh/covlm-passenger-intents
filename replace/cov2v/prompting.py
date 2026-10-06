from __future__ import annotations

import math
from pathlib import Path

from .schema import CooperativeObservation, VehicleState, command_to_intent

IMAGE_INPUT_ORDER = ("left", "front", "right")


def build_system_prompt(template_path: Path | None = None) -> str:
    if template_path and template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return (
        "<OBJECTIVE_AND_PERSONA>\n"
        "You are a cautious autopilot assistant for cooperative driving.\n"
        "</OBJECTIVE_AND_PERSONA>\n\n"
        "<GLOBAL_CONTEXT>\n"
        "The ego vehicle receives left, front, and right camera images.\n"
        "Scenario descriptions identify cooperative vehicles and provide their communicated intentions.\n"
        "Camera images may contain both cooperative vehicles and non-communicating traffic participants, such as pedestrians and other vehicles.\n"
        "Use Scenario to know which visible vehicles are cooperative, and use images to recognize those cooperative vehicles when visible and to infer non-communicating participants.\n"
        "</GLOBAL_CONTEXT>\n\n"
        "<GLOBAL_CONSTRAINTS>\n"
        "Do not invent hidden vehicles, hidden intentions, or unobserved participants.\n"
        "Do not treat a visually observed actor as cooperative unless it is listed in Scenario.\n"
        "Do not use cooperative messages as evidence for unlisted non-communicating traffic participants.\n"
        "</GLOBAL_CONSTRAINTS>"
    )


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _section(tag: str, content: str) -> str:
    return f"<{tag}>\n{content.strip()}\n</{tag}>"


def _format_scalar(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_block(value, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_format_block(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{prefix}- None"
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(_format_block(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.append(_format_block(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{_format_scalar(value)}"


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1))


def _json_output_format(contract: dict) -> str:
    return (
        "Return exactly one valid JSON object. Do not wrap it in markdown. "
        "Use exactly these fields and enum strings:\n"
        f"{_format_block(contract)}"
    )


def _image_input_instruction() -> str:
    return (
        "Attached image inputs follow this exact order: left camera image, "
        "front camera image, right camera image."
    )


def _task_text(stage: str, objective: str, section_guidance: list[str]) -> str:
    stage_text = str(stage).strip()
    if stage_text and stage_text[-1] not in ".!?":
        stage_text += "."
    return "\n".join(
        [
            f"You are now doing this part of the task: {stage_text}",
            f"Your requirement: {objective}",
            "Use only the current scene. Read and use the following sections as instructed:",
            _numbered(section_guidance),
        ]
    )


def _position_phrase(forward: float, right: float) -> str:
    parts = []
    if abs(forward) < 0.5:
        parts.append("nearly level with you longitudinally")
    elif forward > 0.0:
        parts.append(f"{abs(forward):.1f} m ahead of you")
    else:
        parts.append(f"{abs(forward):.1f} m behind you")

    if abs(right) < 1.5:
        parts.append("near your centerline laterally")
    elif right > 0.0:
        parts.append(f"{abs(right):.1f} m to your right")
    else:
        parts.append(f"{abs(right):.1f} m to your left")
    return " and ".join(parts)


def _heading_relation(ego_theta: float, other_theta: float) -> str:
    angle_diff = _wrap_angle(other_theta - ego_theta)
    if -math.pi / 4 <= angle_diff <= math.pi / 4:
        return "heading forward relative to ego"
    if math.pi / 4 < angle_diff <= 3 * math.pi / 4:
        return "heading right relative to ego"
    if -3 * math.pi / 4 <= angle_diff < -math.pi / 4:
        return "heading left relative to ego"
    return "heading backward relative to ego"


def _driver_intent_phrase(ego_state: VehicleState) -> str:
    intent = ego_state.driver_intent
    if intent is None:
        return ""

    parts = [f"driver_intent is {intent.intent_type}"]
    if intent.urgency_level:
        parts.append(f"urgency level is {int(intent.urgency_level)}")
    if intent.requested_priority_bonus:
        parts.append(f"requested priority bonus is {int(intent.requested_priority_bonus)}")
    justification = str(intent.justification).strip()
    if justification:
        parts.append(f"justification: {justification}")
    return ", and " + ", ".join(parts)


def _build_scenario(ego_state: VehicleState, cooperative_observations: list[CooperativeObservation]) -> str:
    lines = [
        "**Scenario**",
        "- Relative positions for surrounding vehicles use the vehicle center for longitudinal ahead/behind distance, and the closest point on that vehicle body to ego's forward center axis for lateral left/right distance.",
        (
            f"- Ego Vehicle (ID: {ego_state.vehicle_id}): "
            f"current driving intention is {command_to_intent(ego_state.command)}, "
            f"estimated driving intention in about 3 seconds is {command_to_intent(ego_state.future_command_3s)}, "
            f"speed = {round(ego_state.speed, 1)} m/s"
            f"{_driver_intent_phrase(ego_state)}"
        ),
        "- Surrounding Cooperative Vehicles:",
    ]
    if not cooperative_observations:
        lines.append("  - None within the current cooperative communication range.")
        return "\n".join(lines)

    for obs in cooperative_observations:
        forward, right = obs.relative_position_ego
        lines.append(
            "  - Vehicle (ID: {vehicle_id}): {relation}, {heading}, "
            "speed = {speed:.1f} m/s, current driving intention is {intent}, "
            "and estimated driving intention in about 3 seconds is {future_intent}.".format(
                vehicle_id=obs.other_id,
                relation=_position_phrase(forward, right),
                heading=_heading_relation(ego_state.pose.theta, obs.heading_theta_global),
                speed=float(obs.speed),
                intent=str(obs.command_intent),
                future_intent=str(obs.future_command_3s_intent),
            )
        )
    return "\n".join(lines)


def _serialize_ego_vehicle(ego_state: VehicleState) -> dict:
    payload = {
        "vehicle_id": ego_state.vehicle_id,
        "speed_mps": round(ego_state.speed, 1),
        "route_intent": command_to_intent(ego_state.command),
        "future_route_intent_3s": command_to_intent(ego_state.future_command_3s),
    }
    if ego_state.driver_intent:
        payload["driver_intent"] = {
            "intent_type": ego_state.driver_intent.intent_type,
            "urgency_level": ego_state.driver_intent.urgency_level,
            "requested_priority_bonus": ego_state.driver_intent.requested_priority_bonus,
            "justification": ego_state.driver_intent.justification,
        }
    return payload


def _serialize_priority_for_discussion(priority: dict) -> dict:
    return {
        "vehicle_id": str(priority.get("vehicle_id", "")),
        "conflict_detected": bool(priority.get("conflict_detected", False)),
        "priority_score": int(priority.get("priority_score", 0)),
        "priority_reason": str(priority.get("priority_reason", "")),
        "potential_conflict_vehicle_ids": [
            str(vehicle_id) for vehicle_id in priority.get("potential_conflict_vehicle_ids", [])
        ],
    }


def _summarize_turn(turn: dict) -> dict:
    actions = []
    for item in turn.get("proposed_actions", []):
        vehicle_id = str(item.get("vehicle_id", "")).strip()
        target_speed_action = str(item.get("target_speed_action", "")).strip()
        if vehicle_id and target_speed_action:
            actions.append(f"{vehicle_id}:{target_speed_action}")
    return {
        "round": int(turn.get("round_index", 0)),
        "speaker": str(turn.get("speaker_id", "")),
        "agrees": bool(turn.get("agree_with_existing_plan", False)),
        "actions": ", ".join(actions),
        "message": str(turn.get("message", "")).strip(),
    }


def _driving_judgment_chain(include_strategy_knowledge: bool = True) -> list[str]:
    if not include_strategy_knowledge:
        return [
            "Describe the scenario in detail in your mind based on the input images information first.",
            "Understand the ego route_intent, speed, and visible scene context.",
            "Use Scenario to identify available cooperative-vehicle observations when they are provided.",
            "Identify direct safety risks from observations and images.",
            "Choose one target_speed_action by following DECISION_GUIDE and the output format.",
        ]
    return [
        "Describe the scenario in detail in your mind based on the input images information first.",
        "Understand the road direction, ego route_intent, speed, and cooperative-vehicle relations.",
        "Use Scenario to identify cooperative vehicles, then use images to recognize visible cooperative vehicles and any non-communicating actors.",
        "Identify the key risk: pedestrian, obstacle, front vehicle, stop-sign requirement, merge/lane-change/turn conflict, or route crossing.",
        "Apply traffic rules and safe distance before cooperation preferences.",
        "Use cooperative messages and priority to decide who proceeds and who yields.",
        "When an action is required, choose one target_speed_action by following DECISION_GUIDE.",
    ]


def _action_selection_guide(include_strategy_knowledge: bool = True) -> list[str]:
    if not include_strategy_knowledge:
        return [
            "TARGET_0 means target_speed 0.0 m/s.",
            "TARGET_SLOW means a low rolling target speed, about 2.0 m/s.",
            "TARGET_NORMAL means the normal route target speed. It does not mean keeping the current speed.",
            "TARGET_FAST means faster than the normal route target speed.",
            "Choose exactly one target_speed_action according to the current observations and the required output format.",
        ]
    return [
        "TARGET_0 means target_speed 0.0 m/s with brake. Use only for immediate collision risk, required full stop, complete yielding to a direct conflict, like when turn left ego vehicle is yielding at intersections, or when ego is yielding, already low-speed (0.0-2.0 m/s), and close to the vehicle it must yield to (0.0-12 m). A visible STOP sign can require TARGET_0 only until ego has completed the stop; if yielding ego is already stopped and the intersection/path is clear or have save distance with other vehicles, choose TARGET_NORMAL or TARGET_SLOW for more efficent driving instead of remaining TARGET_0.",
        "TARGET_SLOW means a low rolling target speed, about 2.0 m/s. Use for efficient rolling yield and safe gap creation when ego still has enough distance or speed to keep moving safely. Do not use TARGET_SLOW for a yielding left turn ego vehicle, or for a yielding ego that is already low-speed and close to the vehicle it must yield to, use TARGET_0 instead.",
        "TARGET_NORMAL means normal route target speed. It does not mean keeping the current speed. If current speed is 0 but the vehicle should proceed normally when safe, choose TARGET_NORMAL rather than TARGET_0.",
        "TARGET_FAST means faster than normal route target. Use only when the path is clear, safe distance is sufficient, no pedestrian/non-cooperative risk is visible, and acceleration helps a safe plan. Do not use TARGET_FAST to force a merge, pass through a small gap, or approach a slower front vehicle too quickly.",
        "If evidence is uncertain, choose the safer lower-speed action among TARGET_NORMAL, TARGET_SLOW, and TARGET_0; do not choose TARGET_0 unless the stopping condition is actually met.",
    ]


def _format_decision_guide(*parts) -> str:
    lines = []
    for part in parts:
        if not part:
            continue
        if isinstance(part, str):
            lines.append(part)
        else:
            lines.extend(str(item) for item in part)
    return _numbered(lines)


def _traffic_rules() -> list[str]:
    return [
        "Emergency rule: vehicles with justified special circumstances may pass first if it is safe.",
        "A STOP sign requires a complete stop, not indefinite waiting. After ego has stopped at the stop line/intersection, it may proceed when the path is clear and there is no pedestrian, vehicle, or red-light conflict.",
        "Merging vehicles normally slow down and yield to straight-going vehicles, but may take priority in justified emergency situations if passing is safe.",
        "When side-by-side vehicles have a lane-change conflict, the vehicle that is longitudinally ahead should proceed first, and the vehicle behind should yield unless emergency priority or safety evidence requires otherwise.",
        "Left-turn vehicles normally stop and yield to straight or right-turn vehicles, but may take priority in justified emergency situations if passing is safe.",
        "A straight-going ego vehicle should yield to turn left of turn right vehicle, when other vehicle is located ahead of the ego vehicle and near the ego vehicle's centerline laterally.",
        "The vehicle being yielded to should go faster only when the path is clear and safe.",
        "Vehicles behind must decrease speed during emergency braking only when they are close enough or closing fast enough to create rear-end risk.",
    ]


def _reference_rules(
    include_priority: bool = False,
    include_strategy_knowledge: bool = True,
) -> dict:
    rules = {
        "judgment_chain": _driving_judgment_chain(include_strategy_knowledge),
    }
    if include_strategy_knowledge:
        rules["traffic_rules"] = _traffic_rules()
    if include_priority and include_strategy_knowledge:
        rules["priority_policy"] = _priority_policy()
    return rules


def _priority_policy() -> dict:
    return {
        "definition": "priority_score is the vehicle's current urgency and safe decision authority in the cooperative conflict.",
        "main_rule": "Score only the current scene. Higher priority should speak earlier and carry more weight, but cannot force an unsafe action.",
        "score_bands": {
            "0-20": "No real conflict, far from conflict, or should clearly yield.",
            "21-45": "Possible conflict; coordination useful but not urgent.",
            "46-70": "Clear conflict or near conflict zone; active coordination needed.",
            "71-100": "Immediate/safety-critical conflict, closest to conflict, strong lawful priority, or justified emergency intent.",
        },
        "judgment_factors": [
            "Urgency and safety risk.",
            "Closeness to the conflict point or lane-change/merge zone.",
            "Lawful right-of-way and current path occupancy.",
            "Route_intent conflict with other vehicles or visible actors.",
            "ego_vehicle.driver_intent from config: urgency_level/requested_priority_bonus can strongly increase ego priority when justified by the scene.",
        ],
        "driver_intent_rule": "Use only ego_vehicle.driver_intent as a bounded priority signal for the current ego vehicle. It can change speaking order and yielding expectations when justified, but not safety or traffic-law decisions.",
    }


def _discussion_turn_summaries(transcript: list[dict]) -> list[dict]:
    return [_summarize_turn(turn) for turn in transcript]


def build_negotiation_user_prompt(
    ego_state: VehicleState,
    cooperative_observations: list[CooperativeObservation],
    include_strategy_knowledge: bool = True,
) -> str:
    reference_instruction = (
        "REFERENCE_RULES: apply the listed traffic rules before choosing an action."
        if include_strategy_knowledge
        else "REFERENCE_RULES: use the listed judgment chain only; rely on observations and direct safety evidence without extra strategy rules."
    )
    decision_standards = [
        "Decide only for the ego vehicle in Scenario.",
        "Use Scenario for cooperative-vehicle identities and communicated intentions; use images to verify visible cooperative vehicles, road context, and non-communicating actors.",
        "Resolve safety risks before considering cooperative preference.",
        "If cooperative messages conflict with image evidence or safety, follow the safer interpretation.",
    ]
    output_contract = {
        "target_speed_action": "TARGET_0 | TARGET_SLOW | TARGET_NORMAL | TARGET_FAST, selected using DECISION_GUIDE",
        "reason": "Explain the decision based on your selected target speed action, observed risks, and cooperative context",
        "key_risks": ["Risk points you consider important for this decision"],
    }
    return "\n\n".join(
        [
            _section(
                "TASK",
                _task_text(
                    "single-vehicle decision in CoV2V cooperative driving. This is used when the current ego vehicle needs an individual final target_speed_action outside a discussion group.",
                    "Decide the final target_speed_action for the current ego vehicle only.",
                    [
                        "OBSERVATIONS: read the image order and Scenario to understand ego intent, speed, and cooperative-vehicle relations.",
                        reference_instruction,
                        "DECISION_GUIDE: follow the single action-selection logic for TARGET_0, TARGET_SLOW, TARGET_NORMAL, and TARGET_FAST.",
                        "OUTPUT_FORMAT: return exactly the requested JSON object.",
                    ],
                ),
            ),
            _section(
                "OBSERVATIONS",
                "\n".join(
                    [
                        _image_input_instruction(),
                        _build_scenario(ego_state, cooperative_observations),
                    ]
                ),
            ),
            _section(
                "REFERENCE_RULES",
                _format_block(_reference_rules(include_strategy_knowledge=include_strategy_knowledge)),
            ),
            _section(
                "DECISION_GUIDE",
                _format_decision_guide(
                    "Decision space: TARGET_0, TARGET_SLOW, TARGET_NORMAL, TARGET_FAST.",
                    decision_standards,
                    _action_selection_guide(include_strategy_knowledge),
                ),
            ),
            _section("OUTPUT_FORMAT", _json_output_format(output_contract)),
        ]
    )


def build_priority_user_prompt(
    ego_state: VehicleState,
    cooperative_observations: list[CooperativeObservation],
    include_strategy_knowledge: bool = True,
) -> str:
    reference_instruction = (
        "REFERENCE_RULES: use priority_policy to score urgency, right-of-way, closeness, route intent, and justified driver_intent."
        if include_strategy_knowledge
        else "REFERENCE_RULES: use the listed judgment chain only; score from direct scene evidence without extra priority-policy rules."
    )
    decision_standards = [
        "Set conflict_detected true only when ego has a current or near-term interaction with a communicating vehicle.",
        "Assign priority_score by urgency, closeness to conflict, route_intent, and justified driver_intent.",
        "Use images to recognize visible cooperative vehicles and non-communicating actors as safety context, but list only Scenario cooperative vehicle_ids in potential_conflict_vehicle_ids.",
        "When evidence is weak or the vehicles can proceed independently, use conflict_detected false, score 0-20, and an empty conflict list.",
    ]
    output_contract = {
        "vehicle_id": ego_state.vehicle_id,
        "conflict_detected": "boolean",
        "priority_score": "Integer 0-100; higher means higher urgency and stronger lawful/safe decision priority",
        "priority_reason": "Explain why this priority score is appropriate for the current scene",
        "conflict_summary": "Summarize the cooperative conflict according to your judgment",
        "potential_conflict_vehicle_ids": [
            "List only vehicle_ids from surrounding cooperative vehicles that are potential conflict participants"
        ],
    }
    return "\n\n".join(
        [
            _section(
                "TASK",
                _task_text(
                    "priority assessment in CoV2V cooperative driving. This happens before group discussion and determines whether this ego vehicle has a cooperative conflict and how much priority it should receive.",
                    "Evaluate this ego vehicle's cooperative conflict status and decision priority.",
                    [
                        "OBSERVATIONS: use image order and Scenario to understand ego, cooperative vehicles, and visible risks.",
                        reference_instruction,
                        "DECISION_GUIDE: decide whether a real cooperative conflict exists and which vehicle IDs participate.",
                        "OUTPUT_FORMAT: return exactly the requested JSON object.",
                    ],
                ),
            ),
            _section(
                "OBSERVATIONS",
                "\n".join(
                    [
                        _image_input_instruction(),
                        _build_scenario(ego_state, cooperative_observations),
                    ]
                ),
            ),
            _section(
                "REFERENCE_RULES",
                _format_block(
                    _reference_rules(
                        include_priority=True,
                        include_strategy_knowledge=include_strategy_knowledge,
                    )
                ),
            ),
            _section("DECISION_GUIDE", _numbered(decision_standards)),
            _section("OUTPUT_FORMAT", _json_output_format(output_contract)),
        ]
    )


def build_discussion_turn_user_prompt(
    ego_state: VehicleState,
    cooperative_observations: list[CooperativeObservation],
    own_priority: dict,
    all_priorities: list[dict],
    speaking_order: list[str],
    transcript: list[dict],
    round_index: int,
    include_strategy_knowledge: bool = True,
) -> str:
    reference_instruction = (
        "REFERENCE_RULES: apply the listed traffic rules before agreeing or objecting."
        if include_strategy_knowledge
        else "REFERENCE_RULES: use the listed judgment chain only; rely on observations and direct safety evidence without extra strategy rules."
    )
    if include_strategy_knowledge:
        discussion_decision_guide = [
            "Read previous_turn_summaries carefully, especially each vehicle's message and proposed_actions.",
            "Treat earlier higher-priority vehicles' safe proposals as the default plan to support. Cooperate when their proposal is lawful, safe, and does not conflict with your visual evidence.",
            "Use your own OBSERVATIONS and camera images to check for safety hazards that other vehicles may not see, including visible cooperative vehicles, pedestrians, non-communicating vehicles, blockers, red lights, narrow gaps, or a dangerous front vehicle.",
            "If you see no concrete hidden hazard or traffic-rule conflict, agree with the existing safe plan and explain your support as needed.",
            "Disagree only when you can name a specific safety risk, traffic-rule violation, or image-based evidence conflict. In that case, explain the reason and propose safer replacement actions.",
            "Make the group plan safe first and efficient second: use TARGET_SLOW for efficient rolling yield when there is still enough distance, but use TARGET_0 when a yielding vehicle is already low-speed and close to the yielded vehicle or conflict path.",
            "Return one proposed action for every discussion-group vehicle, not only for yourself.",
            "For every proposed action, use this action-selection logic:",
            *_action_selection_guide(include_strategy_knowledge),
        ]
    else:
        discussion_decision_guide = [
            "Read previous_turn_summaries and the current Scenario.",
            "Use OBSERVATIONS and camera images to identify direct safety risks.",
            "Return one proposed action for every discussion-group vehicle, not only for yourself.",
            "For every proposed action, use this action-selection logic:",
            *_action_selection_guide(include_strategy_knowledge),
        ]
    output_contract = {
        "speaker_id": ego_state.vehicle_id,
        "agree_with_existing_plan": "boolean",
        "proposed_actions": [
            {
                "vehicle_id": "veh_x",
                "target_speed_action": "TARGET_0 | TARGET_SLOW | TARGET_NORMAL | TARGET_FAST, selected using DECISION_GUIDE",
                "short_reason": "Explain why this action is appropriate for that vehicle",
            }
        ],
        "message": "According to your decision, explain whether you support the current action plan or propose a safer action plan",
    }
    other_priorities = [
        _serialize_priority_for_discussion(priority)
        for priority in all_priorities
        if str(priority.get("vehicle_id", "")) != ego_state.vehicle_id
    ]
    return "\n\n".join(
        [
            _section(
                "TASK",
                _task_text(
                    "one vehicle's discussion turn in CoV2V cooperative driving. Priorities and speaking order have already been computed; now this speaker must respond to the current group proposal.",
                    "Produce one cooperative discussion turn for the current speaker in Scenario and seek a mutually safe, efficient group plan.",
                    [
                        "OBSERVATIONS: read the current speaker's Scenario from its ego perspective.",
                        "DISCUSSION_CONTEXT: use speaking order, priorities, and previous turn summaries to understand what other vehicles proposed and why.",
                        reference_instruction,
                        "DECISION_GUIDE: first try to cooperate with other vehicles' safe decisions, then check your own images for hazards they may not see, and only object with a concrete safer proposal.",
                        "OUTPUT_FORMAT: return exactly the requested JSON object.",
                    ],
                ),
            ),
            _section(
                "OBSERVATIONS",
                "\n".join(
                    [
                        _image_input_instruction(),
                        _build_scenario(ego_state, cooperative_observations),
                    ]
                ),
            ),
            _section(
                "DISCUSSION_CONTEXT",
                "\n".join(
                    [
                        f"Round index: {round_index}",
                        "Speaking order:",
                        _format_block(speaking_order),
                        "",
                        "Ego priority:",
                        _format_block(_serialize_priority_for_discussion(own_priority)),
                        "",
                        "Other vehicle priorities:",
                        _format_block(other_priorities),
                        "",
                        "Previous turn summaries:",
                        _format_block(_discussion_turn_summaries(transcript)),
                    ]
                ),
            ),
            _section(
                "REFERENCE_RULES",
                _format_block(_reference_rules(include_strategy_knowledge=include_strategy_knowledge)),
            ),
            _section(
                "DECISION_GUIDE",
                _numbered(discussion_decision_guide),
            ),
            _section("OUTPUT_FORMAT", _json_output_format(output_contract)),
        ]
    )


def build_priority_fallback_user_prompt(
    ego_state: VehicleState,
    cooperative_observations: list[CooperativeObservation],
    own_priority: dict,
    all_priorities: list[dict],
    speaking_order: list[str],
    transcript: list[dict],
    include_strategy_knowledge: bool = True,
) -> str:
    reference_instruction = (
        "REFERENCE_RULES: apply the listed traffic rules before making final actions."
        if include_strategy_knowledge
        else "REFERENCE_RULES: use the listed judgment chain only; rely on observations and direct safety evidence without extra strategy rules."
    )
    if include_strategy_knowledge:
        fallback_decision_guide = [
            "You are the highest-priority speaker. The previous discussion did not produce identical proposed_actions, so you must make the final fallback proposal.",
            "Use previous_turn_summaries to understand every vehicle's last message and proposed_actions. Respect useful safety concerns raised by lower-priority vehicles.",
            "Use your own OBSERVATIONS and camera images to check for hazards that other vehicles may not see.",
            "Choose final_actions for every discussion-group vehicle. The plan must be safe first and efficient second.",
            "If a lower-priority vehicle proposed a safer action because of a concrete visual hazard or traffic-rule risk, incorporate that safer action.",
            "For every final action, use this action-selection logic:",
            *_action_selection_guide(include_strategy_knowledge),
        ]
    else:
        fallback_decision_guide = [
            "You are the fallback speaker because the previous discussion did not produce identical proposed_actions.",
            "Use previous_turn_summaries, OBSERVATIONS, and camera images to identify direct safety risks.",
            "Choose final_actions for every discussion-group vehicle.",
            "For every final action, use this action-selection logic:",
            *_action_selection_guide(include_strategy_knowledge),
        ]
    output_contract = {
        "final_actions": [
            {
                "vehicle_id": "veh_x",
                "target_speed_action": "TARGET_0 | TARGET_SLOW | TARGET_NORMAL | TARGET_FAST, selected using DECISION_GUIDE",
                "reason": "Explain why this final action is appropriate for that vehicle",
                "key_risks": ["Risk points you consider important for this final action"],
            }
        ],
        "fallback_summary": "According to your final decision, summarize why the fallback action plan is safe and efficient",
    }
    other_priorities = [
        _serialize_priority_for_discussion(priority)
        for priority in all_priorities
        if str(priority.get("vehicle_id", "")) != ego_state.vehicle_id
    ]
    return "\n\n".join(
        [
            _section(
                "TASK",
                _task_text(
                    "highest-priority fallback proposal in CoV2V cooperative driving. Discussion turns did not agree on the same actions; now the highest-priority vehicle must make the final action proposal.",
                    "Produce final_actions for every discussion-group vehicle using the discussion history, your ego-view observations, priorities, and available reference rules.",
                    [
                        "OBSERVATIONS: read the highest-priority speaker's Scenario from its ego perspective.",
                        "DISCUSSION_CONTEXT: use priorities, speaking order, and previous turn summaries to understand why actions disagreed.",
                        reference_instruction,
                        "DECISION_GUIDE: make one final safe and efficient action plan for all vehicles.",
                        "OUTPUT_FORMAT: return exactly the requested JSON object.",
                    ],
                ),
            ),
            _section(
                "OBSERVATIONS",
                "\n".join(
                    [
                        _image_input_instruction(),
                        _build_scenario(ego_state, cooperative_observations),
                    ]
                ),
            ),
            _section(
                "DISCUSSION_CONTEXT",
                "\n".join(
                    [
                        "Speaking order:",
                        _format_block(speaking_order),
                        "",
                        "Highest-priority speaker:",
                        _format_block(_serialize_priority_for_discussion(own_priority)),
                        "",
                        "Other vehicle priorities:",
                        _format_block(other_priorities),
                        "",
                        "Previous turn summaries:",
                        _format_block(_discussion_turn_summaries(transcript)),
                    ]
                ),
            ),
            _section(
                "REFERENCE_RULES",
                _format_block(_reference_rules(include_strategy_knowledge=include_strategy_knowledge)),
            ),
            _section(
                "DECISION_GUIDE",
                _numbered(fallback_decision_guide),
            ),
            _section("OUTPUT_FORMAT", _json_output_format(output_contract)),
        ]
    )
