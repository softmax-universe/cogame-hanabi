## Actual native batched client, parser normalization, and authoritative apply.
import std/[json, options, os]
import bitworld/decision_trajectory
import hanabi/[llm, sim, training, prompts]

putEnv("COWORLD_LLM_ENDPOINT", paramStr(1))
putEnv("COWORLD_LLM_MODEL", "checkpoint/native-fixture")
putEnv("COWORLD_LLM_TEMPERATURE", "0")
var config = defaultGameConfig()
config.sampled = true
config.hostedModel = paramStr(3)
config.players = @[PlayerConfig(name: "a"), PlayerConfig(name: "b"),
  PlayerConfig(name: "c"), PlayerConfig(name: "d")]
config.tokens = @["a", "b", "c", "d"]
var game = initSim(config)
let before = game
let operatorPrompts = @["Exact private operator prompt", "", "", ""]
let scripts = @[skNone, skNone, skNone, skNone]
let scenario = paramStr(2)
let decision = newLlmClient(config).decideAll(game, @[0], operatorPrompts, scripts, PromptSelection(variant: pvMixed, seed: 4))[0]
if scenario == "infra":
  doAssert decision.nativeAttempts.len == 1
  doAssert not decision.nativeAttempts[0].accepted
  doAssert decision.origin == "missing"
  doAssert decision.failureKind == fkInfra
  doAssert decision.nativeAttempts[0].httpStatus == some(503)
else:
  doAssert decision.nativeAttempts.len == 2
  doAssert not decision.nativeAttempts[0].accepted
  if scenario == "retry":
    doAssert decision.origin == "retry"
    doAssert decision.failureKind == fkNone
    doAssert decision.nativeAttempts[1].accepted
    doAssert sameMove(decision.move, Move(kind: akPlay, slot: 1, target: -1))
  else:
    doAssert decision.origin == "rejected"
    doAssert decision.failureKind == fkModel
    doAssert not decision.nativeAttempts[1].accepted
  for attempt in decision.nativeAttempts:
    doAssert attempt.platformCallId.isSome
    doAssert attempt.rawResponse.kind == JObject
    doAssert attempt.request["temperature"].getFloat() == 0
    doAssert attempt.latencyMs.isSome
    doAssert attempt.inputTokens.get() == 3
    doAssert attempt.outputTokens.get() == 2
let trajectory = newDecisionTrajectory("probe", "probe", "hanabi", "fixture", "fixture-source")
if scenario == "retry":
  game.applyMove(0, decision.move, decision.note, decision.banner, decision.origin)
  trajectory.recordAppliedDecision(before, 0, decision, decision, operatorPrompts[0], game.done, PromptSelection(variant: pvMixed, seed: 4))
  trajectory.finish(esTruncated, game.resultsJson(), %*{})
else:
  doAssert not game.done
  doAssert game.turn == 0
  let isInfra = decision.failureKind == fkInfra
  trajectory.recordRejectedDecision(before, 0, decision, operatorPrompts[0], PromptSelection(variant: pvMixed, seed: 4))
  trajectory.finish(esFailed, %*{
    "error": true,
    "status": "failed",
    "failure_category": (if isInfra: "infrastructure_error" else: "rejected_action"),
    "action_status": (if isInfra: "missing" else: "rejected"),
    "seat": 0,
    "turn": 0,
    "reason": decision.reject
  }, newJObject())
echo trajectory.eventsJsonl()
