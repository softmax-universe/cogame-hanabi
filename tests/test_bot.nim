## The scripted baselines must play whole episodes without ever proposing an
## illegal move — they are the no-credentials fallback (offline
## certification), the fallback for a rejected model reply, and fieldable
## policies. And the reply parser has to be tolerant in exactly the ways the
## design note lists, and strict everywhere else.

import std/[monotimes, strutils, times, unicode, unittest]
import hanabi/[llm, sim, prompts]

proc fixture(seed: int, maxTurns = 80): GameConfig =
  result = defaultGameConfig()
  result.seed = seed
  result.maxTurns = maxTurns
  result.sampled = true
  for index in 0 ..< Seats:
    result.players.add(PlayerConfig(name: "P" & $(index + 1)))
    result.tokens.add("t" & $index)

proc playScripted(config: GameConfig, kinds: array[Seats, ScriptKind]): Sim =
  result = initSim(config)
  while not result.done:
    let seat = result.pendingSeats()[0]
    let decision = scriptedAction(result, seat, kinds[seat])
    ## The proposal must be legal AS IS: the legal-move list is the rule.
    check result.illegalReason(decision.move).len == 0
    var found = false
    for move in result.legalMoves():
      if sameMove(move, decision.move):
        found = true
    check found
    check decision.note.len == 0
    check decision.banner.len == 0
    result.applyMove(seat, decision.move, "", "", "scripted")

suite "scripted baselines":
  test "both baselines play 200 seeded episodes legally and fast":
    var slowest = 0
    for seed in 1 .. 200:
      for kinds in [[skConventions, skConventions, skConventions,
          skConventions],
        [skCautious, skCautious, skCautious, skCautious],
        [skConventions, skCautious, skConventions, skCautious]]:
        let started = getMonoTime()
        let sim = playScripted(fixture(seed), kinds)
        let elapsed = (getMonoTime() - started).inMilliseconds.int
        slowest = max(slowest, elapsed)
        check sim.done
        check sim.reason == "complete"
        check sim.turn <= 80
    echo "slowest scripted episode: ", slowest, " ms"
    when defined(release):
      check slowest < 50

  test "cautious never misplays":
    var fusesLost = 0
    for seed in 1 .. 200:
      let sim = playScripted(fixture(seed),
        [skCautious, skCautious, skCautious, skCautious])
      fusesLost += MaxFuses - sim.fuses
      check sim.endReason != "strikeout"
    check fusesLost == 0

  test "conventions beats cautious and banks half the fireworks":
    var conventions = 0
    var cautious = 0
    for seed in 1 .. 50:
      conventions += playScripted(fixture(seed),
        [skConventions, skConventions, skConventions, skConventions]).score()
      cautious += playScripted(fixture(seed),
        [skCautious, skCautious, skCautious, skCautious]).score()
    let meanConventions = conventions / 50
    let meanCautious = cautious / 50
    echo "mean score: conventions ", meanConventions, ", cautious ",
      meanCautious
    check meanConventions > meanCautious
    check meanConventions >= 12.0

  test "decideAll with no credentials runs scripted seats and rejects non-scripted seats":
    let config = fixture(3, maxTurns = 40)
    let client = newLlmClient(config)
    client.disabled = true
    check client.disabled
    var sim = initSim(config)
    var turns = 0
    while not sim.done and turns < 12:
      let seats = sim.pendingSeats()
      check seats.len == 1
      let seat = seats[0]
      let scripted = [skNone, skCautious, skNone, skConventions]
      let decisions = client.decideAll(sim, seats, @["be bold", "", "", ""],
        @scripted, PromptSelection())
      check decisions.len == 1
      if scripted[seat] == skNone:
        check decisions[0].origin == "rejected"
        check decisions[0].reject.len > 0
        ## Advance with a legal move only to check the next seat in this loop.
        sim.applyMove(seat, scriptedAction(sim, seat, skConventions).move, "", "", "scripted")
      else:
        check sameMove(decisions[0].move, scriptedAction(sim, seat, scripted[seat]).move)
        check decisions[0].origin == "scripted"
        sim.applyMove(seat, decisions[0].move, "", "", decisions[0].origin)
      turns += 1
    check turns == 12

suite "reply parsing":
  proc parsed(sim: Sim, text: string): Decision =
    parseDecision(sim, extractJsonObject(text))

  test "canonical JSON moves parse bare or inside a single outer markdown code fence":
    var sim = initSim(fixture(5))
    sim.hands[1].cards[0].card = Card(colour: 0, rank: 3)   ## a red 3
    sim.hands[1].cards[1].card = Card(colour: 2, rank: 1)
    sim.hands[1].cards[2].card = Card(colour: 3, rank: 4)
    sim.hands[1].cards[3].card = Card(colour: 4, rank: 2)
    let wantColour = Move(kind: akHint, target: 1, hintKind: hkColour, value: 0)
    check sameMove(sim.parsed(
      """{"action":"hint","target":1,"hintType":"colour","hintValue":"red"}"""
      ).move, wantColour)
    ## Single outer markdown code fence: json-tagged or plain.
    check sameMove(sim.parsed("```json\n{\"action\":\"play\",\"slot\":2}\n```")
      .move, Move(kind: akPlay, slot: 2, target: -1))
    check sameMove(sim.parsed("```\n{\"action\":\"play\",\"slot\":3}\n```")
      .move, Move(kind: akPlay, slot: 3, target: -1))
    check sameMove(sim.parsed("```JSON\r\n{\"action\":\"discard\",\"slot\":4}\r\n```")
      .move, Move(kind: akDiscard, slot: 4, target: -1))
    ## Rank hint with canonical string hintValue.
    let rank = Move(kind: akHint, target: 1, hintKind: hkRank, value: 3)
    check sameMove(sim.parsed(
      """{"action":"hint","target":1,"hintType":"rank","hintValue":"3"}"""
      ).move, rank)
    ## Optional note and banner fields still work.
    let withExtras = sim.parsed(
      """{"action":"play","slot":1,"note":"keep","banner":"go"}""")
    check sameMove(withExtras.move, Move(kind: akPlay, slot: 1, target: -1))
    check withExtras.note == "keep"
    check withExtras.banner == "go"

  test "notes and banners are capped on rune boundaries":
    var sim = initSim(fixture(5))
    var long = ""
    for index in 0 ..< 700:
      long.add("é")
    let decision = sim.parsed("""{"action":"play","slot":1,"note":"""" &
      long & """","banner":"""" & long & """"}""")
    check decision.note.runeLen == MaxNoteLen
    check decision.banner.runeLen == MaxBannerLen
    check decision.note.validateUtf8() == -1
    check decision.banner.validateUtf8() == -1
    check cleanText(long, MaxNoteLen).runeLen == MaxNoteLen
    check cleanText(long, MaxBannerLen).runeLen == MaxBannerLen
    ## Newlines never survive into a banner.
    let multi = sim.parsed(
      """{"action":"play","slot":1,"banner":"one\ntwo"}""")
    check multi.banner == "one two"

  test "legacy shims, aliases, prose, extra objects, and bad fields are rejected":
    var sim = initSim(fixture(5))
    let rejectedReplies = [
      """{"action":"ponder","slot":1}""",
      """{"note":"no action here"}""",
      "I would play slot 1, I think.",
      """{"move":"play 2"}""",
      """{"move":"hint 1 red"}""",
      """{"action":"tell","target":1,"hintType":"colour","hintValue":"red"}""",
      """{"action":"clue","target":1,"hintType":"colour","hintValue":"red"}""",
      """{"action":"hint","target":1,"hintType":"color","hintValue":"red"}""",
      """{"action":"hint","target":1,"hintType":"suit","hintValue":"red"}""",
      """{"action":"hint","target":1,"hintType":"number","hintValue":"3"}""",
      """{"action":"hint","target":1,"hintType":"value","hintValue":"3"}""",
      """{"action":"hint","target":1,"hintType":"colour","hintValue":"r"}""",
      """{"action":"hint","target":1,"hint":"red"}""",
      """{"action":"hint","target":1,"hintValue":"red"}""",
      """{"action":"hint","target":1,"hintType":"rank","hintValue":3}""",
      """{"action":"hint","target":"""" & sim.names[1] & """","hintType":"colour","hintValue":"red"}""",
      """{"action":"hint","target":"1","hintType":"colour","hintValue":"red"}""",
      """{"action":"play","slot":"2"}""",
      """{"action":"play","slot":2.0}""",
      """{"action":"play","slot":1,"extra":true}""",
      """{"action":"play","slot":1,"target":2}""",
      """{"action":"play","slot":1,"note":123}""",
      """{"action":"play","slot":1,"banner":[]}""",
      """{"action": "play", "slot": 2} — I think that is the red 1.""",
      """Sure: {"action":"play","slot":2}""",
      """{"action":"play","slot":1} {"action":"play","slot":2}""",
      "```json\n{\"action\":\"play\",\"slot\":1}\n```\nExtra prose.",
      "Intro\n```json\n{\"action\":\"play\",\"slot\":1}\n```",
      "```python\n{\"action\":\"play\",\"slot\":1}\n```",
      "```json\n{\"action\":\"play\",\"slot\":1}",
      "```json\n{\"action\":\"play\",\"slot\":1}\n```\n```json\n{\"action\":\"play\",\"slot\":2}\n```",
      """[]""",
      """"play 1""""
    ]
    for reply in rejectedReplies:
      expect HanabiError:
        discard sim.parsed(reply)
    ## Parseable but illegal: the legal-move list is what refuses these.
    check sim.illegalReason(sim.parsed(
      """{"action":"play","slot":0}""").move).len > 0
    check sim.illegalReason(sim.parsed(
      """{"action":"play","slot":9}""").move).len > 0
    check sim.illegalReason(sim.parsed(
      """{"action":"hint","target":0,"hintType":"rank","hintValue":"1"}"""
      ).move).len > 0                                   ## self-hint
    check sim.hintTokens == MaxHintTokens
    check sim.illegalReason(sim.parsed(
      """{"action":"discard","slot":1}""").move).len > 0  ## 8 tokens
    ## An empty hint: nobody at this table holds a green card.
    var empty = initSim(fixture(5))
    for slot in 1 .. HandSize:
      empty.hands[1].cards[slot - 1].card = Card(colour: 0, rank: slot)
    let move = empty.parsed(
      """{"action":"hint","target":1,"hintType":"colour","hintValue":"green"}"""
      ).move
    check empty.illegalReason(move).len > 0
    check "at least one card" in empty.illegalReason(move)

  test "PLAYER_SCRIPTED values map to the two baselines":
    check parseScriptKind("1") == skConventions
    check parseScriptKind("true") == skConventions
    check parseScriptKind("conventions") == skConventions
    check parseScriptKind("cautious") == skCautious
    check parseScriptKind("careful") == skCautious
    check parseScriptKind("") == skNone
    check parseScriptKind("nonsense") == skNone
