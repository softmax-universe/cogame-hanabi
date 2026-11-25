"""Public host + native player WebSocket round-trip tests.

Exercises the real compiled `/bin/hanabi` host and `/bin/hanabi-player`
binaries across the certification seat mix:
  1. Regular JSON + Markdown code fences + incidental private `note` (verified
     end-to-end with `tools/verify_native_posttrain.nim`).
  2. Invalid / ambiguous model object -> unscored `rejected_action` failure, no
     `results.json`, no numeric scores.
  3. Provider HTTP 503 failure -> unscored `infrastructure_error` failure, no
     `results.json`, no numeric scores.
  4. Missing LLM credentials with a native model seat -> unscored
     `infrastructure_error` failure, no `results.json`, no numeric scores.
"""

import json
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "coworld_manifest_template.json").read_text())


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_healthz(port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/healthz"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(f"host did not become healthy on port {port}")


def extract_first_legal_move(user_content: str) -> dict:
    match = re.search(r"^\s*1\.\s*(\{.*\})\s*$", user_content, re.MULTILINE)
    assert match, f"no legal move found in prompt:\n{user_content[-500:]}"
    return json.loads(match.group(1))


def start_mock_llm(mode: str):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/messages"
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            call_id = str(uuid.uuid4())
            turn_index = len(calls)
            calls.append((req, call_id))

            if mode == "provider_503":
                body = b"service temporarily unavailable"
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Softmax-Llm-Call-Id", call_id)
                self.end_headers()
                self.wfile.write(body)
                return

            if mode == "ambiguous":
                raw = '{"action":"play","slot":1} {"action":"play","slot":2}'
            else:
                move = extract_first_legal_move(req["messages"][0]["content"])
                if turn_index == 0:
                    # Regular bare JSON with an incidental private note & banner.
                    move["note"] = "slot 1 note on first turn"
                    move["banner"] = "Opening move"
                    raw = json.dumps(move)
                elif turn_index == 1:
                    # json-tagged Markdown fence without a new note (verifies persistent note handling).
                    raw = f"```json\n{json.dumps(move)}\n```"
                elif turn_index == 2:
                    # Plain Markdown fence with a new note.
                    move["note"] = "follow-up note"
                    raw = f"```\n{json.dumps(move)}\n```"
                else:
                    raw = json.dumps(move)

            body = json.dumps(
                {
                    "id": call_id,
                    "model": req.get("model", "anthropic/claude-haiku-4.5"),
                    "content": [{"type": "text", "text": raw}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 12, "output_tokens": 8},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Softmax-Llm-Call-Id", call_id)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, calls


def run_episode(
    hanabi_bin: Path,
    player_bin: Path,
    work_dir: Path,
    llm_endpoint: str | None,
    episode_id: str,
    source_rev: str,
):
    port = free_port()
    cert = MANIFEST["certification"]
    config = dict(cert["game_config"])
    config["maxTurns"] = 20
    config["turnDelayMs"] = 0
    config["tokens"] = [f"tok-{i}" for i in range(4)]
    config_path = work_dir / "config.json"
    config_path.write_text(json.dumps(config))

    results_path = work_dir / "results.json"
    replay_path = work_dir / "replay.json"
    failure_path = work_dir / "player_failure.json"
    trajectory_path = work_dir / "trajectory.jsonl"

    game_env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "COWORLD_LLM_ENDPOINT",
            "COWORLD_LLM_MODEL",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_API_KEY_URI",
            "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
            "AWS_BEARER_TOKEN_BEDROCK",
            "BEDROCK_MODEL",
        }
    }
    game_env.update(MANIFEST["game"]["runnable"].get("env", {}))
    game_env.update(
        {
            "COGAME_HOST": "127.0.0.1",
            "COGAME_PORT": str(port),
            "COGAME_CONFIG_URI": f"file://{config_path}",
            "COGAME_RESULTS_URI": f"file://{results_path}",
            "COGAME_SAVE_REPLAY_URI": f"file://{replay_path}",
            "COGAME_PLAYER_FAILURE_URI": f"file://{failure_path}",
            "COGAME_SAVE_TRAJECTORY_URI": f"file://{trajectory_path}",
            "COWORLD_EPISODE_ID": episode_id,
            "COWORLD_GAME_VERSION": "test-0.1",
            "COWORLD_SOURCE_REVISION": source_rev,
        }
    )
    if llm_endpoint:
        game_env["COWORLD_LLM_ENDPOINT"] = llm_endpoint

    host = subprocess.Popen(
        [str(hanabi_bin)],
        cwd=ROOT,
        env=game_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    players = []
    try:
        wait_healthz(port)
        by_id = {entry["id"]: entry for entry in MANIFEST["player"]}
        for slot, seat_spec in enumerate(cert["players"]):
            entry = by_id[seat_spec["player_id"]]
            penv = dict(os.environ)
            penv.pop("PLAYER_SCRIPTED", None)
            penv.pop("PLAYER_PROMPT", None)
            penv.update(entry.get("env", {}))
            penv["COWORLD_PLAYER_WS_URL"] = (
                f"ws://127.0.0.1:{port}/player?slot={slot}&token=tok-{slot}"
            )
            proc = subprocess.Popen(
                [str(player_bin)],
                cwd=ROOT,
                env=penv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            players.append(proc)

        host_out, _ = host.communicate(timeout=45)
        player_codes = []
        for proc in players:
            proc.communicate(timeout=10)
            player_codes.append(proc.returncode)
        return (
            host.returncode,
            host_out,
            player_codes,
            results_path,
            replay_path,
            failure_path,
            trajectory_path,
        )
    finally:
        if host.poll() is None:
            host.kill()
            host.wait(timeout=5)
        for proc in players:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


def assert_unscored_failure(
    rc: int,
    results_path: Path,
    replay_path: Path,
    failure_path: Path,
    trajectory_path: Path,
    expected_category: str,
    expected_action_status: str,
):
    assert rc != 0, "failed episode must exit non-zero"
    assert not results_path.exists(), "results.json must not be written on failure"
    assert not replay_path.exists(), "replay.json must not be written on failure"
    assert failure_path.exists(), "player_failure.json must be written on failure"
    failure = json.loads(failure_path.read_text())
    assert failure["error"] is True
    assert failure["status"] == "failed"
    assert failure["failure_category"] == expected_category
    assert failure["action_status"] == expected_action_status
    for forbidden in ("scores", "score", "winner", "fireworks"):
        assert forbidden not in failure, f"forbidden key {forbidden} in {failure}"

    assert trajectory_path.exists(), "trajectory.jsonl must record failure evidence"
    records = [json.loads(line) for line in trajectory_path.read_text().splitlines() if line.strip()]
    decision = records[-2]
    summary = records[-1]
    assert decision["event_type"] == "decision"
    assert decision["action_status"] == expected_action_status
    assert decision["selected_attempt_id"] is None
    assert decision["executed_action"] is None
    assert summary["event_type"] == "episode"
    assert summary["status"] == "failed"
    assert summary["participant_outcomes"] == {}
    assert summary["outcome"]["failure_category"] == expected_category
    assert summary["outcome"]["action_status"] == expected_action_status
    for forbidden in ("scores", "score", "winner", "fireworks"):
        assert forbidden not in summary["outcome"]


def main():
    # Verify manifest certification seats every advertised player ID and configures native LLM env.
    declared_ids = [entry["id"] for entry in MANIFEST["player"]]
    assert declared_ids[0] == "hanabi-player"
    assert "PLAYER_SCRIPTED" not in MANIFEST["player"][0].get("env", {})
    assert MANIFEST["player"][0]["env"].get("COWORLD_LLM_ENABLED") == "true"
    assert MANIFEST["game"]["runnable"]["env"].get("COWORLD_LLM_ENABLED") == "true"
    cert_ids = {seat["player_id"] for seat in MANIFEST["certification"]["players"]}
    assert set(declared_ids).issubset(cert_ids), f"missing declared IDs: {set(declared_ids) - cert_ids}"

    with tempfile.TemporaryDirectory(prefix="hanabi-roundtrip-") as tmp:
        tmp_path = Path(tmp)
        nim = os.environ.get("NIM", "nim")
        hanabi_bin = tmp_path / "hanabi"
        player_bin = tmp_path / "hanabi-player"
        verify_bin = tmp_path / "verify_native_posttrain"
        for src, out in [
            ("src/hanabi.nim", hanabi_bin),
            ("src/hanabi_player.nim", player_bin),
            ("tools/verify_native_posttrain.nim", verify_bin),
        ]:
            subprocess.run(
                [
                    nim,
                    "c",
                    "--hints:off",
                    "--path:src",
                    f"--out:{out}",
                    f"--nimcache:{tmp_path / ('nimcache-' + out.name)}",
                    src,
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

        # 1. Valid round-trip: bare JSON + markdown fences + incidental private note.
        server, thread, calls = start_mock_llm("valid")
        try:
            ep_dir = tmp_path / "valid"
            ep_dir.mkdir()
            rc, out, player_codes, results_path, replay_path, failure_path, traj_path = (
                run_episode(
                    hanabi_bin,
                    player_bin,
                    ep_dir,
                    f"http://127.0.0.1:{server.server_port}",
                    "ep-valid",
                    "rev-test",
                )
            )
            assert rc == 0, out
            assert all(code == 0 for code in player_codes), player_codes
            assert len(calls) >= 3, f"expected >=3 native LLM calls, got {len(calls)}"
            assert not failure_path.exists()
            results = json.loads(results_path.read_text())
            assert results["reason"] == "complete"
            assert len(results["scores"]) == 4
            assert len(set(results["scores"])) == 1
            replay = json.loads(replay_path.read_text())
            for ev in replay["events"]:
                if ev.get("kind") == "move":
                    assert "text" not in ev, "private note leaked into public replay"
            # Run verify_native_posttrain on replay + trajectory (with incidental notes!).
            verify = subprocess.run(
                [str(verify_bin), str(replay_path), str(traj_path), "ep-valid", "rev-test"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            assert verify.returncode == 0, verify.stderr + verify.stdout
            print("roundtrip valid (JSON + fences + note + verify_native_posttrain): PASS")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)

        # 2. Invalid / ambiguous model object -> unscored rejected_action failure.
        server, thread, calls = start_mock_llm("ambiguous")
        try:
            ep_dir = tmp_path / "ambiguous"
            ep_dir.mkdir()
            rc, out, player_codes, results_path, replay_path, failure_path, traj_path = (
                run_episode(
                    hanabi_bin,
                    player_bin,
                    ep_dir,
                    f"http://127.0.0.1:{server.server_port}",
                    "ep-ambiguous",
                    "rev-test",
                )
            )
            assert len(calls) == 2, f"expected 2 attempts (initial + retry), got {len(calls)}"
            assert_unscored_failure(
                rc, results_path, replay_path, failure_path, traj_path,
                "rejected_action", "rejected",
            )
            print("roundtrip ambiguous object (unscored rejected_action): PASS")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)

        # 3. Provider HTTP 503 failure -> unscored infrastructure_error failure.
        server, thread, calls = start_mock_llm("provider_503")
        try:
            ep_dir = tmp_path / "provider_503"
            ep_dir.mkdir()
            rc, out, player_codes, results_path, replay_path, failure_path, traj_path = (
                run_episode(
                    hanabi_bin,
                    player_bin,
                    ep_dir,
                    f"http://127.0.0.1:{server.server_port}",
                    "ep-503",
                    "rev-test",
                )
            )
            assert len(calls) == 1, f"expected 1 attempt on provider 503, got {len(calls)}"
            assert_unscored_failure(
                rc, results_path, replay_path, failure_path, traj_path,
                "infrastructure_error", "missing",
            )
            print("roundtrip provider 503 (unscored infrastructure_error): PASS")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)

        # 4. Missing credentials with native model seat -> unscored infrastructure_error.
        ep_dir = tmp_path / "no_creds"
        ep_dir.mkdir()
        rc, out, player_codes, results_path, replay_path, failure_path, traj_path = (
            run_episode(
                hanabi_bin,
                player_bin,
                ep_dir,
                None,
                "ep-nocreds",
                "rev-test",
            )
        )
        assert_unscored_failure(
            rc, results_path, replay_path, failure_path, traj_path,
            "infrastructure_error", "missing",
        )
        print("roundtrip missing credentials (unscored infrastructure_error): PASS")


if __name__ == "__main__":
    main()
