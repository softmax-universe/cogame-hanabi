## Claude-backed decision making for Hanabi. Each seat's policy is just a
## prompt: the game server composes the seat's view (the partners' hands,
## its own hand as knowledge only, the public log, the enumerated legal
## moves, its notes) plus that seat's prompt and asks Claude what it does.
##
## Hanabi is TURN-BASED: seat `turn mod 4` acts alone and its observation
## depends on the move made on the turn before, so decisions cannot be
## batched. The starter's batching code is kept unchanged anyway —
## `pendingSeats` returns a one-element seq, so `decideAll` issues
## `curly.makeRequests` with a batch of ONE and the retry path, the parsing
## and the fallback are the same code on a shorter list.
##
## Credentials, in order of preference:
##   COWORLD_LLM_ENDPOINT            - hosted sidecar
##   Bedrock bearer token            - local play
##   ANTHROPIC_API_KEY                - the key itself
##   ANTHROPIC_API_KEY_URI            - a URI holding the key
## Explicit scripted seats run their deterministic baseline without an LLM.
## Non-scripted seats without credentials or with failed/rejected inference
## return a failure decision rather than substituting a scripted move.

import
  std/[json, options, os, strutils, times, unicode],
  bitworld/runtime,
  bitworld/decision_trajectory,
  curly,
  sim, prompts

const
  AnthropicUrl = "https://api.anthropic.com/v1/messages"
  AnthropicVersion = "2023-06-01"
  BedrockAnthropicVersion = "bedrock-2023-05-31"
  DefaultBedrockModel* = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
  ## The hosted Bedrock sidecar caps 30 requests a minute per episode and a
  ## throttle cascades, so consecutive requests START at least this far
  ## apart. A turn is one request (two with a retry), so this bounds the
  ## episode at 30/minute by construction whatever the latency.
  MinRequestSpacingSeconds* = 2.0
  ## Caps on rejected-reply diagnostics, in RUNES.
  MaxRejectLen* = 200

type
  LlmInfraError* = object of HanabiError

  FailureKind* = enum
    fkNone
    fkModel
    fkInfra

  ScriptKind* = enum
    skNone = "none"
    skConventions = "conventions"
    skCautious = "cautious"

  Decision* = object
    move*: Move
    note*: string       ## private to the seat, fed back next turn
    banner*: string     ## spectator-only, never read by any seat
    origin*: string     ## llm | retry | external | scripted | rejected | missing
    policy*: string     ## actual scripted teacher identity
    reject*: string     ## why the model's reply was refused or unavailable
    failureKind*: FailureKind
    nativeAttempts*: seq[DecisionAttempt]

  LlmTransport = enum
    ltNone, ltSidecar, ltBedrock, ltAnthropic

  LlmClient* = ref object
    curl: Curly
    transport: LlmTransport
    apiKey: string          ## anthropic transport
    sidecarEndpoint: string
    bedrockEndpoint: string ## local Bedrock transport
    bedrockModel*: string   ## single configured Bedrock model (no fallback rotation)
    bedrockToken: string
    model: string         ## active model id
    maxOutputTokens: int
    timeoutSeconds: int
    lastRequestAt: float  ## start of the most recent batch, for the floor
    disabled*: bool   ## true once credentials are known-unavailable

proc parseScriptKind*(text: string): ScriptKind =
  ## PLAYER_SCRIPTED values: "1"/"true"/"yes"/"conventions" play the
  ## convention-following bot, "cautious" the never-misplays bot, anything
  ## else nothing.
  case text.strip().toLowerAscii()
  of "1", "true", "yes", "conventions", "convention": skConventions
  of "cautious", "careful": skCautious
  else: skNone

proc resolveApiKey(): string =
  result = getEnv("ANTHROPIC_API_KEY").strip()
  if result.len > 0:
    return
  let uri = getEnv("ANTHROPIC_API_KEY_URI").strip()
  if uri.len == 0:
    return ""
  try:
    result = readCogameUri(uri, "ANTHROPIC_API_KEY_URI").strip()
  except CatchableError as error:
    echo "hanabi llm: failed to fetch ANTHROPIC_API_KEY_URI: ", error.msg
    result = ""

proc bedrockModelId*(): string =
  ## Single Bedrock inference-profile id. BEDROCK_MODEL pins a custom id;
  ## otherwise use DefaultBedrockModel with no implicit model fallback rotation.
  let pinned = getEnv("BEDROCK_MODEL").strip()
  if pinned.len > 0:
    return pinned
  DefaultBedrockModel

proc bedrockUrl*(client: LlmClient): string =
  client.bedrockEndpoint & "/model/" & client.bedrockModel & "/invoke"

proc newLlmClient*(config: GameConfig): LlmClient =
  result = LlmClient(
    model: config.model,
    maxOutputTokens: config.maxOutputTokens,
    timeoutSeconds: config.llmTimeoutSeconds
  )
  let sidecarEndpoint = getEnv("COWORLD_LLM_ENDPOINT").strip()
  if config.hostedModel.len > 0 and sidecarEndpoint.len == 0:
    raise newException(HanabiError, "hostedModel requires COWORLD_LLM_ENDPOINT")
  if sidecarEndpoint.len > 0:
    result.transport = ltSidecar
    result.sidecarEndpoint = sidecarEndpoint.strip(chars = {'/'}, leading = false)
    result.model = if config.hostedModel.len > 0: config.hostedModel
      else: getEnv("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5")
    result.curl = newCurly()
    return
  let bedrockEndpoint = getEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME").strip()
  let bedrockToken = getEnv("AWS_BEARER_TOKEN_BEDROCK").strip()
  if bedrockEndpoint.len > 0 or bedrockToken.len > 0:
    let region = getEnv("AWS_REGION",
      getEnv("AWS_DEFAULT_REGION", "us-west-2"))
    let endpoint =
      if bedrockEndpoint.len > 0: bedrockEndpoint
      else: "https://bedrock-runtime." & region & ".amazonaws.com"
    result.transport = ltBedrock
    result.bedrockEndpoint = endpoint.strip(chars = {'/'}, leading = false)
    result.bedrockModel = bedrockModelId()
    result.model = result.bedrockModel
    result.bedrockToken = bedrockToken
    result.curl = newCurly()
    echo "hanabi llm: bedrock transport, model ",
      result.bedrockModel,
      ", url ", result.bedrockUrl
    return
  result.apiKey = resolveApiKey()
  if result.apiKey.len > 0:
    result.transport = ltAnthropic
    result.curl = newCurly()
    echo "hanabi llm: anthropic transport, model ", result.model
  else:
    result.transport = ltNone
    result.disabled = true
    echo "hanabi llm: no LLM credentials; non-scripted seats will fail unscored"

# ---- Scripted baselines -----------------------------------------------------
#
# Both are pure functions of the sim and the acting seat, both are
# deterministic, and neither ever proposes a move outside `legalMoves`.

proc actsIn(seat, other: int): int =
  (other - seat + Seats) mod Seats

proc hintKey(seat: int, move: Move, primary: int): (int, int, int, int, int) =
  ## Tie-breaks, in order: the primary score, then the seat acting soonest,
  ## then rank before colour, then the lower rank / earlier colour, then the
  ## lower seat index.
  (primary, actsIn(seat, move.target),
    (if move.hintKind == hkRank: 0 else: 1), move.value, move.target)

proc conventionsMove*(sim: Sim, seat: int): Move =
  ## The strong, convention-following partner a prompt has to beat. The
  ## first rule that fires wins.
  let view = sim.view()
  let moves = sim.legalMoves()
  if moves.len == 0:
    return Move(kind: akPlay, slot: 1, target: -1)
  let basis = view.basisFor(seat)

  ## 1. Play the lowest-numbered slot that is known playable.
  for slot in 1 .. view.hands[seat].size:
    if view.knownPlayableWith(basis, seat, slot):
      return Move(kind: akPlay, slot: slot, target: -1)

  ## 2. Save: the next seat to act is about to lose the last copy of a card.
  let next = (seat + 1) mod Seats
  if sim.hintTokens >= 1:
    let chop = view.chopSlot(next)
    if chop >= 1:
      let card = view.hands[next].cards[chop - 1].card
      if card.rank >= 1 and view.isCriticalIdent(card.colour, card.rank):
        let save = Move(kind: akHint, target: next, hintKind: hkRank,
          value: card.rank)
        if sim.illegalReason(save).len == 0:
          return save

  ## 3. Play-hint: the hint that makes the most of the receiver's cards
  ## NEWLY known-playable, at least one.
  var bestHint: Move
  var bestKey = (0, 0, 0, 0, 0)
  var haveHint = false
  for move in moves:
    if move.kind != akHint:
      continue
    let gained = view.annotateView(sim.names, move, sim.turn).nowPlayable.len
    if gained < 1:
      continue
    let key = hintKey(seat, move, -gained)
    if not haveHint or key < bestKey:
      bestKey = key
      bestHint = move
      haveHint = true
  if haveHint:
    return bestHint

  ## 4. Discard the chop (which already prefers a known-dead slot).
  if sim.hintTokens < MaxHintTokens:
    let chop = view.chopSlot(seat)
    if chop >= 1:
      return Move(kind: akDiscard, slot: chop, target: -1)

  ## 5. Stall hint: touch as little as possible, preferring the next seat.
  var haveStall = false
  var stall: Move
  var stallKey = (0, 0, 0, 0, 0, 0)
  for move in moves:
    if move.kind != akHint:
      continue
    let touched = view.hintTouches(move.target, move.hintKind, move.value).len
    let base = hintKey(seat, move, touched)
    let key = ((if move.target == next: 0 else: 1), base[0], base[1], base[2],
      base[3], base[4])
    if not haveStall or key < stallKey:
      stallKey = key
      stall = move
      haveStall = true
  if haveStall:
    return stall

  ## 6. Unreachable — kept so the function is total.
  moves[0]

proc cautiousMove*(sim: Sim, seat: int): Move =
  ## A deliberately simpler partner: the unfamiliar teammate of the
  ## cross-play story. It never misplays.
  let view = sim.view()
  let moves = sim.legalMoves()
  if moves.len == 0:
    return Move(kind: akPlay, slot: 1, target: -1)
  let basis = view.basisFor(seat)

  ## 1. Play only what it can name: a singleton candidate set that is
  ## playable right now.
  for slot in 1 .. view.hands[seat].size:
    let cards = view.candidatesWith(basis, seat, slot)
    if cards.len == 1 and view.isPlayable(cards[0].colour, cards[0].rank):
      return Move(kind: akPlay, slot: slot, target: -1)

  ## 2. Tell somebody the rank of a playable card they know nothing about.
  if sim.hintTokens >= 1:
    for step in 1 ..< Seats:
      let target = (seat + step) mod Seats
      for slot in 1 .. view.hands[target].size:
        let held = view.hands[target].cards[slot - 1]
        if held.hintColour >= 0 or held.hintRank > 0:
          continue
        if not view.isPlayable(held.card.colour, held.card.rank):
          continue
        let move = Move(kind: akHint, target: target, hintKind: hkRank,
          value: held.card.rank)
        if sim.illegalReason(move).len == 0:
          return move

  ## 3. Discard the chop.
  if sim.hintTokens < MaxHintTokens:
    let chop = view.chopSlot(seat)
    if chop >= 1:
      return Move(kind: akDiscard, slot: chop, target: -1)

  ## 4. Stall: the rank of the next seat's oldest card.
  for step in 1 ..< Seats:
    let target = (seat + step) mod Seats
    let size = view.hands[target].size
    if size == 0:
      continue
    let move = Move(kind: akHint, target: target, hintKind: hkRank,
      value: view.hands[target].cards[size - 1].card.rank)
    if sim.illegalReason(move).len == 0:
      return move

  moves[0]

proc scriptedAction*(sim: Sim, seat: int, kind: ScriptKind): Decision =
  ## Rule-based baseline for `seat`. Always legal; never notes or banners.
  result.origin = "scripted"
  result.policy = "scripted-" & $kind
  # Partners see our cards, but we cannot use their resulting card-count
  # deductions without knowing those cards ourselves. Choose from the same
  # information boundary as the seat's private observation.
  var visible = sim
  for slot in 0 ..< visible.hands[seat].size:
    visible.hands[seat].cards[slot].card = Card(colour: 0, rank: 0)
  visible.deck = newSeq[Card](sim.deck.len)
  visible.events = @[]
  visible.config.seed = 0
  for other in 0 ..< Seats:
    if other != seat:
      visible.notes[other] = ""
      visible.banners[other] = ""
  result.move =
    if kind == skCautious: cautiousMove(visible, seat)
    else: conventionsMove(visible, seat)

# ---- Prompt building --------------------------------------------------------

const CanonicalRules = """

The deck: 50 cards, five colours (red, yellow, green, blue, white), each
with ranks 1 1 1 2 2 3 3 4 4 5 — three 1s, two each of 2, 3 and 4, and a
single 5.

Your hand: four cards, held FACING OUT. Everyone else can see your cards;
YOU CANNOT. You can see everyone else's. Slots are numbered 1..4 left to
right, slot 1 is the newest card and the highest-numbered slot the oldest.
When a card leaves your hand you draw a new one into slot 1 and the others
shift one slot higher; once the deck is empty nothing is drawn and your hand
gets shorter.

The table: five firework stacks, one per colour, each starting at 0 and
built strictly 1, 2, 3, 4, 5. A public discard pile. EIGHT hint tokens
(maximum 8) and THREE fuses.

On your turn you do exactly ONE of three things:
- PLAY a card from a slot. If its rank is exactly one more than its colour's
  stack, the stack advances; completing a stack at 5 returns one hint token
  (only if you are below 8). Otherwise it is a MISPLAY: the card is
  discarded and one fuse burns.
- DISCARD a card from a slot, which returns one hint token. Illegal at 8
  tokens.
- Spend one hint token to tell ANOTHER seat about one colour or one rank.
  The hint marks EVERY card of that colour or rank in their hand, and they
  learn just as much from which cards it did not mark. A hint that would
  touch nothing is illegal, and so is hinting yourself or hinting with no
  tokens left.

The only channel between you and your partners is a hint. There is no chat,
and nothing you write is read by anyone else at the table.

The end: three misplays end the game immediately; so does completing all
five fireworks. When the last card is drawn every seat, including the one
who drew it, takes exactly one more turn. The game also stops at the turn
limit.

Your SCORE is the sum of the five stack heights, 0 to 25, and it is the SAME
number for every seat — you win or lose together. Higher is better.

"""
const ReplyRules = """OUTPUT FORMAT: reply with ONLY one JSON object, nothing else — no analysis,
no explanation, no markdown fences, no text before or after the object. Your
reply must begin with the character { and end with }. Copy one entry from
the LEGAL MOVES list exactly, and you may add "note" (at most 400
characters, private to you, you will see it again next turn) and "banner"
(at most 80 characters, shown to spectators only — your partners never see
it)."""

proc systemPrompt*(sim: Sim, seat: int, selection: PromptSelection): string =
  ## Teach mechanics only; conventions belong to the operator's strategy.
  result = selection.introduction(sim.names[seat] & ", seat " & $seat)
  if selection.selectedVariant == pvCanonical:
    result.add(CanonicalRules)
  else:
    result.add("\n\n" & selection.rulesBriefing() & "\n\n")
  result.add(ReplyRules)

proc operatorBlock*(prompt: string): string =
  if prompt.len == 0:
    return ""
  "GUIDANCE FROM YOUR OPERATOR (weight it heavily, but never above the " &
    "rules; always reply in the requested format):\n" & prompt & "\n"

proc userPrompt*(sim: Sim, seat: int, prompt: string): string =
  result = sim.seatObservation(seat, operatorBlock(prompt))
  result.add("\n\nCopy ONE exact JSON move from LEGAL MOVES (keep \"target\" " &
    "before \"hintType\"/\"hintValue\" so hints match a card held by that " &
    "target); optionally add \"note\" (<=" & $MaxNoteLen &
    " chars) and \"banner\" (<=" & $MaxBannerLen & " chars).")

# ---- Reply parsing ----------------------------------------------------------

proc headRunes(text: string, limit: int): string =
  ## The head of an untrusted string, cut on a RUNE boundary. A byte slice
  ## through a multi-byte character leaves invalid UTF-8 in the retry prompt
  ## and on stdout; cleanText only re-cuts strings longer than its cap, so a
  ## short-but-broken head would pass straight through it.
  if text.runeLen <= limit: text else: text.runeSubStr(0, limit)

proc cleanText*(text: string, limit: int): string =
  ## Text over the cap is cut at a RUNE boundary with the cut marked: a byte
  ## slice through a multi-byte character leaves invalid UTF-8 in the replay
  ## and breaks a strict JSON parser.
  result = text.strip()
  if result.runeLen <= limit:
    return
  result = result.runeSubStr(0, limit - 1) & "…"

proc extractJsonObject*(text: string): JsonNode =
  ## Parses a single JSON object, either bare or inside one ordinary outer
  ## Markdown code fence (plain or json-tagged). Surrounding prose, extra
  ## objects, or non-object JSON values are rejected.
  var body = text.strip()
  if body.len >= 3 and body[0 .. 2] == "\xEF\xBB\xBF":
    body = body[3 .. ^1].strip()
  if body.startsWith("```"):
    let firstBreak = body.find('\n')
    if firstBreak < 0:
      raise newException(HanabiError, "unclosed markdown code fence")
    let tag = body[3 ..< firstBreak].strip().toLowerAscii()
    if tag notin ["", "json"]:
      raise newException(HanabiError,
        "markdown code fence must be plain or json-tagged")
    var inner = body[firstBreak + 1 .. ^1]
    if not inner.endsWith("```"):
      raise newException(HanabiError, "unclosed markdown code fence")
    inner = inner[0 ..< inner.len - 3]
    let lastNl = inner.rfind('\n')
    if lastNl < 0 or inner[lastNl + 1 .. ^1].strip().len > 0:
      raise newException(HanabiError,
        "closing markdown code fence must be on its own line")
    body = inner.strip()
  if body.len == 0:
    raise newException(HanabiError, "empty response")
  try:
    result = parseJson(body)
  except JsonParsingError as error:
    raise newException(HanabiError,
      "response must be a single JSON object: " & error.msg)
  if result.kind != JObject:
    raise newException(HanabiError, "response must be a single JSON object")

proc parseDecision*(sim: Sim, payload: JsonNode): Decision =
  ## Parses a canonical move JSON object matching LEGAL MOVES, plus optional
  ## string "note" and "banner" fields, or raises HanabiError. Legality is
  ## checked by the caller via `illegalReason`.
  if payload.isNil or payload.kind != JObject:
    raise newException(HanabiError, "reply must be a JSON object")
  if payload.hasKey("note") and payload["note"].kind != JNull:
    if payload["note"].kind != JString:
      raise newException(HanabiError, "\"note\" must be a string")
    result.note = cleanText(payload["note"].getStr(), MaxNoteLen)
  if payload.hasKey("banner") and payload["banner"].kind != JNull:
    if payload["banner"].kind != JString:
      raise newException(HanabiError, "\"banner\" must be a string")
    result.banner = cleanText(payload["banner"].getStr(), MaxBannerLen)
      .replace("\n", " ").replace("\r", " ")
  if not payload.hasKey("action") or payload["action"].kind != JString:
    raise newException(HanabiError,
      "no \"action\" in the reply: it must be play, discard or hint")
  let action = payload["action"].getStr()
  case action
  of "play", "discard":
    for key in payload.keys:
      if key notin ["action", "slot", "note", "banner"]:
        raise newException(HanabiError,
          "unexpected field \"" & key & "\" for " & action)
    if not payload.hasKey("slot") or payload["slot"].kind != JInt:
      raise newException(HanabiError, "\"slot\" must be an integer")
    result.move = Move(
      kind: (if action == "play": akPlay else: akDiscard),
      slot: payload["slot"].getInt(),
      target: -1
    )
  of "hint":
    for key in payload.keys:
      if key notin ["action", "target", "hintType", "hintValue", "note", "banner"]:
        raise newException(HanabiError,
          "unexpected field \"" & key & "\" for hint")
    if not payload.hasKey("target") or payload["target"].kind != JInt:
      raise newException(HanabiError, "\"target\" must be an integer seat index")
    let target = payload["target"].getInt()
    if not payload.hasKey("hintType") or payload["hintType"].kind != JString:
      raise newException(HanabiError,
        "\"hintType\" must be \"colour\" or \"rank\"")
    let hintType = payload["hintType"].getStr()
    if not payload.hasKey("hintValue") or payload["hintValue"].kind != JString:
      raise newException(HanabiError,
        "a hint needs string \"hintValue\": a colour name or rank \"1\"..\"5\"")
    let hintValue = payload["hintValue"].getStr()
    case hintType
    of "colour":
      let colour = ColourNames.find(hintValue)
      if colour < 0:
        raise newException(HanabiError,
          "\"" & hintValue & "\" is not one of red, yellow, green, blue, white")
      result.move = Move(kind: akHint, target: target, hintKind: hkColour,
        value: colour)
    of "rank":
      const Ranks = ["1", "2", "3", "4", "5"]
      let rankIndex = Ranks.find(hintValue)
      if rankIndex < 0:
        raise newException(HanabiError,
          "a rank hint is \"1\"..\"5\", not \"" & hintValue & "\"")
      result.move = Move(kind: akHint, target: target, hintKind: hkRank,
        value: rankIndex + 1)
    else:
      raise newException(HanabiError,
        "\"hintType\" must be \"colour\" or \"rank\"")
  else:
    raise newException(HanabiError,
      "\"" & action & "\" is not play, discard or hint")

# ---- Anthropic / Bedrock transport ------------------------------------------

proc requestFor(client: LlmClient, system, user: string, slot: int):
    tuple[url: string, headers: HttpHeaders, body: string] =
  var body = %*{
    "max_tokens": client.maxOutputTokens,
    "system": system,
    "messages": [{"role": "user", "content": user}]
  }
  let temperature = parseFloat(getEnv("COWORLD_LLM_TEMPERATURE", "1"))
  if not (temperature >= 0 and temperature <= 1):
    raise newException(ValueError, "COWORLD_LLM_TEMPERATURE must be finite and between 0 and 1")
  body["temperature"] = %temperature
  var headers: HttpHeaders
  if client.transport == ltSidecar and slot >= 0:
    headers["X-Coworld-Player-Slot"] = $slot
  headers["content-type"] = "application/json"
  if client.transport == ltBedrock:
    body["anthropic_version"] = %BedrockAnthropicVersion
    if client.bedrockToken.len > 0:
      headers["authorization"] = "Bearer " & client.bedrockToken
    result.url = client.bedrockUrl()
  elif client.transport == ltSidecar:
    body["model"] = %client.model
    headers["anthropic-version"] = AnthropicVersion
    result.url = client.sidecarEndpoint & "/v1/messages"
  else:
    body["model"] = %client.model
    ## Only the Claude 5 / Opus tiers accept an effort setting; Haiku 4.5
    ## rejects the whole request with a 400 if it is present.
    if "haiku" notin client.model and "4-5" notin client.model:
      body["output_config"] = %*{"effort": "low"}
    headers["x-api-key"] = client.apiKey
    headers["anthropic-version"] = AnthropicVersion
    result.url = AnthropicUrl
  result.headers = headers
  result.body = $body

proc textOf*(client: LlmClient, response: Response, error, url: string):
    string =
  ## The text of one batched reply, or an error describing why there is none.
  ## Transport, auth, throttle, and non-2xx provider errors raise LlmInfraError
  ## without rotating to an alternate model.
  if error.len > 0:
    raise newException(LlmInfraError, "llm transport: " & error)
  if response.code == 401 or response.code == 403:
    let detail = headRunes(response.body, 400)
    client.disabled = true
    raise newException(LlmInfraError,
      "llm auth failed (" & $response.code & ") at " & url & ": " & detail)
  if response.code == 429:
    let detail = headRunes(response.body, 300)
    raise newException(LlmInfraError, "llm throttled (429): " & detail)
  if response.code < 200 or response.code >= 300:
    raise newException(LlmInfraError, "llm provider error " & $response.code &
      ": " & headRunes(response.body, 300))
  var payload: JsonNode
  try:
    payload = parseJson(response.body)
  except CatchableError as parseErr:
    raise newException(LlmInfraError,
      "llm provider returned invalid JSON: " & parseErr.msg)
  if payload{"stop_reason"}.getStr() == "refusal":
    raise newException(HanabiError, "anthropic refusal")
  if not payload.hasKey("content") or payload["content"].kind != JArray:
    raise newException(LlmInfraError, "llm provider response missing content")
  for contentBlock in payload["content"]:
    if contentBlock{"type"}.getStr() == "text":
      result.add(contentBlock{"text"}.getStr())
  if payload{"stop_reason"}.getStr() == "max_tokens":
    let head = headRunes(result, 160).replace("\n", " ")
    if '{' notin result:
      raise newException(HanabiError, "reply cut off at max_tokens before " &
        "any JSON: " & head)
    ## The object started but the cap landed inside it. Name the cap as the
    ## cause here, so a log line separates a truncated reply from a model
    ## that genuinely emitted malformed JSON.
    try:
      discard extractJsonObject(result)
    except CatchableError:
      raise newException(HanabiError, "reply cut off at max_tokens " &
        "mid-JSON: " & head)

proc spaceRequests(client: LlmClient) =
  ## Hold consecutive request STARTS MinRequestSpacingSeconds apart on shared
  ## remote endpoints; loopback sidecars need no artificial throttle delay.
  if client.transport == ltSidecar and (
      client.sidecarEndpoint.startsWith("http://127.0.0.1") or
      client.sidecarEndpoint.startsWith("http://localhost")):
    return
  let wait = client.lastRequestAt + MinRequestSpacingSeconds - epochTime()
  if wait > 0.0:
    sleep(int(wait * 1000))
  client.lastRequestAt = epochTime()

proc decideAll*(
  client: LlmClient,
  sim: Sim,
  seats: seq[int],
  prompts: seq[string],
  scripted: seq[ScriptKind],
  selection: PromptSelection
): seq[Decision] =
  ## One decision per seat in `seats`, in order — for Hanabi that list is
  ## always the single seat whose turn it is, so this is one request and at
  ## most one retry on a rejected model reply. Genuine scripted seats run
  ## their baseline; non-scripted failures return failureKind=fkInfra
  ## (origin="missing") or failureKind=fkModel (origin="rejected") without
  ## substituting a scripted move.
  result = newSeq[Decision](seats.len)
  var open: seq[int]     ## indexes into `seats` still undecided
  var rejects = newSeq[string](seats.len)
  for index, seat in seats:
    let kind = scripted[seat]
    if kind != skNone:
      result[index] = scriptedAction(sim, seat, kind)
    elif client.disabled:
      result[index] = Decision(
        origin: "missing",
        failureKind: fkInfra,
        reject: "Inference credentials unavailable"
      )
    else:
      open.add(index)
  for attempt in 0 .. 1:
    if open.len == 0 or client.disabled:
      break
    var batch: RequestBatch
    var pendingEvidence: seq[DecisionAttempt]
    for index in open:
      let seat = seats[index]
      var user = sim.userPrompt(seat, prompts[seat])
      if attempt > 0:
        user.add("\n\nYour previous reply was rejected: " &
          cleanText(rejects[index], MaxRejectLen) &
          ". Reply with ONLY one JSON object copied from the LEGAL MOVES " &
          "list.")
      let request = client.requestFor(systemPrompt(sim, seat, selection), user, seat)
      var evidence = newDecisionAttempt("turn-" & $sim.turn & "-attempt-" & $attempt,
        client.model, aoModel)
      evidence.prompt = %*[{"role": "system", "content": systemPrompt(sim, seat, selection)},
        {"role": "user", "content": user}]
      evidence.request = parseJson(request.body)
      evidence.model = some(client.model)
      evidence.decoder = %*{"temperature": (if evidence.request.hasKey("temperature"): evidence.request["temperature"] else: newJNull()),
        "max_tokens": client.maxOutputTokens, "timeout_ms": client.timeoutSeconds * 1000}
      pendingEvidence.add(evidence)
      batch.post(request.url, request.headers, request.body, $index)
    client.spaceRequests()
    let started = epochTime()
    let responses = client.curl.makeRequests(batch, client.timeoutSeconds)
    var stillOpen: seq[int]
    for position, index in open:
      let seat = seats[index]
      var evidence = pendingEvidence[position]
      evidence.latencyMs = some((epochTime() - started) * 1000)
      let nativeResponse = responses[position].response
      if nativeResponse.code >= 100 and nativeResponse.code <= 599:
        evidence.httpStatus = some(nativeResponse.code)
      evidence.rawResponse = %nativeResponse.body
      for (header, field) in [
          ("x-softmax-llm-call-id", "call"),
          ("x-coworld-checkpoint-sha256", "model"),
          ("x-coworld-tokenizer-sha256", "tokenizer"),
          ("x-coworld-chat-template-sha256", "template")]:
        if nativeResponse.headers[header].len > 0:
          case field
          of "call": evidence.platformCallId = some(nativeResponse.headers[header])
          of "model": evidence.modelIdentity = some(nativeResponse.headers[header])
          of "tokenizer": evidence.tokenizerIdentity = some(nativeResponse.headers[header])
          of "template": evidence.chatTemplateSha256 = some(nativeResponse.headers[header])
      var text = ""
      try:
        if nativeResponse.code >= 200 and nativeResponse.code < 300 and responses[position].error.len == 0:
          var payload: JsonNode
          try:
            payload = parseJson(nativeResponse.body)
          except CatchableError as parseErr:
            raise newException(LlmInfraError,
              "llm provider returned invalid JSON: " & parseErr.msg)
          evidence.rawResponse = payload
          if payload.hasKey("model") and payload["model"].kind == JString:
            evidence.model = some(payload["model"].getStr())
          if payload.hasKey("stop_reason") and payload["stop_reason"].kind == JString:
            evidence.stopReason = some(payload["stop_reason"].getStr())
          if payload.hasKey("usage") and payload["usage"].kind == JObject:
            if payload["usage"].hasKey("input_tokens") and payload["usage"]["input_tokens"].kind == JInt:
              evidence.inputTokens = some(payload["usage"]["input_tokens"].getInt())
            if payload["usage"].hasKey("output_tokens") and payload["usage"]["output_tokens"].kind == JInt:
              evidence.outputTokens = some(payload["usage"]["output_tokens"].getInt())
          if payload.hasKey("sampling_evidence") and payload["sampling_evidence"].kind != JNull:
            let sampled = payload["sampling_evidence"]
            var promptTokens, completionTokens: seq[int]
            var probabilities: seq[float]
            for token in sampled["prompt_token_ids"]: promptTokens.add(token.getInt())
            for token in sampled["completion_token_ids"]: completionTokens.add(token.getInt())
            if sampled["behavior_log_probs"].kind != JNull:
              for probability in sampled["behavior_log_probs"]: probabilities.add(probability.getFloat())
            evidence.promptTokenIds = some(promptTokens)
            evidence.sampledTokenIds = some(completionTokens)
            if sampled["behavior_log_probs"].kind != JNull:
              evidence.behaviorLogprobs = some(probabilities)
            evidence.stopReason = some(sampled["stop_reason"].getStr())
        text = client.textOf(responses[position].response,
          responses[position].error, batch[position].url)
        evidence.response = %text
        var decision = parseDecision(sim, extractJsonObject(text))
        ## The legal-move list IS the rule: reject anything outside it here
        ## so the retry can quote exactly what was wrong.
        let reason = sim.illegalReason(decision.move)
        if reason.len > 0:
          raise newException(HanabiError, reason)
        decision.origin = if attempt == 0: "llm" else: "retry"
        evidence.parsedAction = moveJson(decision.move)
        if decision.note.len > 0:
          evidence.parsedAction["note"] = %decision.note
        if decision.banner.len > 0:
          evidence.parsedAction["banner"] = %decision.banner
        evidence.accepted = true
        result[index].nativeAttempts.add(evidence)
        decision.nativeAttempts = result[index].nativeAttempts
        result[index] = decision
      except LlmInfraError as error:
        evidence.response = %text
        evidence.rejectionReason = some(error.msg)
        result[index].nativeAttempts.add(evidence)
        echo "hanabi llm: seat ", seat, " attempt ", attempt, " infrastructure failure"
        let attempts = result[index].nativeAttempts
        result[index] = Decision(
          origin: "missing",
          failureKind: fkInfra,
          reject: cleanText(error.msg, MaxRejectLen),
          nativeAttempts: attempts
        )
      except CatchableError as error:
        evidence.response = %text
        evidence.rejectionReason = some(error.msg)
        result[index].nativeAttempts.add(evidence)
        echo "hanabi llm: seat ", seat, " attempt ", attempt, " rejected"
        rejects[index] = error.msg
        stillOpen.add(index)
    open = stillOpen
  for index in open:
    let seat = seats[index]
    echo "hanabi llm: seat ", seat, " rejected after retry; failing episode unscored"
    let attempts = result[index].nativeAttempts
    result[index] = Decision(
      origin: "rejected",
      failureKind: fkModel,
      reject: cleanText(rejects[index], MaxRejectLen),
      nativeAttempts: attempts
    )
