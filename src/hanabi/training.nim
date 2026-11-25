## Authoritative private language decisions; spectator replay stays separate.
import std/[json, options]
import bitworld/decision_trajectory
import llm, sim, prompts

proc decisionAction*(decision: Decision): JsonNode =
  result = moveJson(decision.move)
  if decision.note.len > 0:
    result["note"] = %decision.note
  if decision.banner.len > 0:
    result["banner"] = %decision.banner

proc recordAppliedDecision*(trajectory: DecisionTrajectory, before: Sim,
    seat: int, proposed, applied: Decision, operatorPrompt: string,
    terminal: bool, selection: PromptSelection) =
  let actualAction = applied.decisionAction()
  var attempts = proposed.nativeAttempts
  var selected = none(string)
  if attempts.len == 0:
    let origin = if applied.origin == "scripted": aoTeacher else: aoUnknown
    var evidence = newDecisionAttempt("turn-" & $before.turn & "-applied",
      if origin == aoTeacher: applied.policy else: "external-hanabi", origin)
    evidence.prompt = %*[{"role": "system", "content": systemPrompt(before, seat, selection)},
      {"role": "user", "content": userPrompt(before, seat, operatorPrompt)}]
    evidence.response = %($actualAction)
    evidence.parsedAction = actualAction
    evidence.accepted = true
    attempts.add(evidence)
  for index in 0 ..< attempts.len:
    if attempts[index].accepted:
      selected = some(attempts[index].attemptId)
  var observation = before.playerFrameJson(seat, true)
  observation["operator_prompt"] = %operatorPrompt
  observation["prompt_selection"] = selection.selectionMetadata()
  var metadata = newJArray()
  for attempt in attempts:
    if attempt.origin in {aoModel, aoTeacher}:
      metadata.add(%*{"attempt_id": attempt.attemptId,
        "prompt_template": selection.promptMetadata(
          attempt.prompt[0]["content"].getStr(), attempt.prompt[1]["content"].getStr())})
  observation["prompt_attempts"] = metadata
  trajectory.recordDecision("hanabi-" & $before.turn, $seat,
    observation, attempts, selected, actualAction,
    asAccepted, terminal = terminal)

proc recordRejectedDecision*(trajectory: DecisionTrajectory, before: Sim,
    seat: int, rejected: Decision, operatorPrompt: string,
    selection: PromptSelection) =
  let status =
    if rejected.failureKind == fkInfra or rejected.origin == "missing":
      asMissing
    else:
      asRejected
  let defaultLabel = if status == asMissing: "missing" else: "rejected"
  var attempts = rejected.nativeAttempts
  if attempts.len == 0:
    var evidence = newDecisionAttempt("turn-" & $before.turn & "-" & defaultLabel,
      "hanabi-" & defaultLabel, aoUnknown)
    evidence.prompt = %*[{"role": "system", "content": systemPrompt(before, seat, selection)},
      {"role": "user", "content": userPrompt(before, seat, operatorPrompt)}]
    evidence.accepted = false
    evidence.rejectionReason = some(
      if rejected.reject.len > 0: rejected.reject else: defaultLabel)
    attempts.add(evidence)
  for index in 0 ..< attempts.len:
    if attempts[index].accepted:
      attempts[index].accepted = false
      attempts[index].rejectionReason = some(
        if rejected.reject.len > 0: rejected.reject else: defaultLabel)
  var observation = before.playerFrameJson(seat, true)
  observation["operator_prompt"] = %operatorPrompt
  observation["prompt_selection"] = selection.selectionMetadata()
  var metadata = newJArray()
  for attempt in attempts:
    if attempt.origin in {aoModel, aoTeacher}:
      metadata.add(%*{"attempt_id": attempt.attemptId,
        "prompt_template": selection.promptMetadata(
          attempt.prompt[0]["content"].getStr(), attempt.prompt[1]["content"].getStr())})
  observation["prompt_attempts"] = metadata
  trajectory.recordDecision("hanabi-" & $before.turn, $seat,
    observation, attempts, none(string), newJNull(),
    status, terminal = true)
