import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import closing
from typing import Dict, Optional, Tuple


DEFAULT_BASE_URL = "http://127.0.0.1:18000/v1"
DEFAULT_MODEL = "Qwen3.5-122B-A10B"


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(0.5)
            try:
                sock.connect((host, port))
                return True
            except OSError:
                time.sleep(0.2)
    return False


def open_ssh_tunnel(ssh_host: str, local_port: int, remote_port: int) -> subprocess.Popen:
    cmd = [
        "ssh",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-N",
        "-L",
        f"{local_port}:127.0.0.1:{remote_port}",
        ssh_host,
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def request_json(
    method: str,
    url: str,
    payload: Optional[Dict] = None,
    timeout_s: float = 30.0,
) -> Tuple[int, Dict]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return int(resp.status), json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw_body": body}
        return int(exc.code), parsed


def terminate_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def build_base_url(
    via_ssh: bool,
    base_url: str,
    ssh_host: str,
    remote_port: int,
    local_port: Optional[int],
    startup_timeout_s: float,
) -> Tuple[str, Optional[subprocess.Popen]]:
    if not via_ssh:
        return base_url.rstrip("/"), None

    chosen_local_port = local_port or find_free_port()
    tunnel = open_ssh_tunnel(
        ssh_host=ssh_host,
        local_port=chosen_local_port,
        remote_port=remote_port,
    )

    if not wait_for_port("127.0.0.1", chosen_local_port, startup_timeout_s):
        stderr = ""
        if tunnel.stderr is not None:
            try:
                stderr = tunnel.stderr.read()
            except Exception:
                stderr = ""
        terminate_process(tunnel)
        raise RuntimeError(f"SSH tunnel failed to start. {stderr.strip()}".strip())

    return f"http://127.0.0.1:{chosen_local_port}/v1", tunnel


def verify_no_key(
    base_url: str,
    via_ssh: bool,
    ssh_host: str,
    model_name: str,
    remote_port: int,
    local_port: Optional[int],
    prompt: str,
    max_tokens: int,
    startup_timeout_s: float,
    request_timeout_s: float,
) -> int:
    tunnel = None

    try:
        resolved_base_url, tunnel = build_base_url(
            via_ssh=via_ssh,
            base_url=base_url,
            ssh_host=ssh_host,
            remote_port=remote_port,
            local_port=local_port,
            startup_timeout_s=startup_timeout_s,
        )

        if via_ssh:
            print(
                f"[info] tunnel ready: {resolved_base_url} -> "
                f"{ssh_host}:127.0.0.1:{remote_port}"
            )
        else:
            print(f"[info] direct mode: {resolved_base_url}")

        print("[info] requests are sent without Authorization header.")

        models_status, models_data = request_json(
            method="GET",
            url=f"{resolved_base_url}/models",
            timeout_s=request_timeout_s,
        )
        print(f"[models] status={models_status}")
        print(json.dumps(models_data, ensure_ascii=False, indent=2))

        if models_status != 200:
            print("[fail] /v1/models did not return 200.", file=sys.stderr)
            return 3

        model_ids = [
            item.get("id")
            for item in models_data.get("data", [])
            if isinstance(item, dict)
        ]
        if model_name not in model_ids:
            print(
                f"[warn] target model {model_name} not found. current models: {model_ids}",
                file=sys.stderr,
            )

        chat_payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        chat_status, chat_data = request_json(
            method="POST",
            url=f"{resolved_base_url}/chat/completions",
            payload=chat_payload,
            timeout_s=request_timeout_s,
        )
        print(f"[chat] status={chat_status}")
        print(json.dumps(chat_data, ensure_ascii=False, indent=2))

        if chat_status != 200:
            print("[fail] /v1/chat/completions did not return 200.", file=sys.stderr)
            return 4

        choices = chat_data.get("choices", [])
        if not choices:
            print("[fail] chat/completions returned no choices.", file=sys.stderr)
            return 5

        print("[pass] no-key access verified.")
        return 0
    finally:
        terminate_process(tunnel)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify whether a vLLM OpenAI-compatible endpoint can be called "
            "without an API key."
        )
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Direct OpenAI-compatible base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.set_defaults(via_ssh=True)
    parser.add_argument(
        "--via-ssh",
        dest="via_ssh",
        action="store_true",
        help="Use an SSH tunnel. This is the current recommended mode.",
    )
    parser.add_argument(
        "--direct",
        dest="via_ssh",
        action="store_false",
        help="Use direct HTTP access instead of SSH tunnel.",
    )
    parser.add_argument(
        "--ssh-host",
        default="AMD-A",
        help="SSH config host alias used when --via-ssh is enabled.",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=18000,
        help="Remote model service port when --via-ssh is enabled.",
    )
    parser.add_argument(
        "--local-port",
        type=int,
        default=18000,
        help="Local forwarded port when --via-ssh is enabled. Default: 18000.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Served model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: NO_KEY_OK",
        help="Prompt used for /v1/chat/completions.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max output tokens for the test completion request.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for SSH tunnel startup.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for each HTTP request.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return verify_no_key(
        base_url=args.base_url,
        via_ssh=args.via_ssh,
        ssh_host=args.ssh_host,
        model_name=args.model,
        remote_port=args.remote_port,
        local_port=args.local_port,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        startup_timeout_s=args.startup_timeout,
        request_timeout_s=args.request_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
