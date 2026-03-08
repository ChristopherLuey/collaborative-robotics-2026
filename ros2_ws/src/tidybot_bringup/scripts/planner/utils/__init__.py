"""Colored terminal logging utilities."""


class C:
    """ANSI color codes."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def log_tool(name: str, args: dict):
    print(f"{C.CYAN}{C.BOLD}⚡ TOOL CALL:{C.RESET} {C.YELLOW}{name}{C.RESET}({C.DIM}{args}{C.RESET})")


def log_result(name: str, result: str):
    preview = result[:200] + '...' if len(result) > 200 else result
    print(f"{C.GREEN}  ✓ {name}:{C.RESET} {preview}")


def log_error(name: str, error: str):
    print(f"{C.RED}  ✗ {name}:{C.RESET} {error}")


def log_gemini(text: str):
    print(f"{C.MAGENTA}{C.BOLD}🤖 Gemini:{C.RESET} {text}")


def log_service(name: str, detail: str = ""):
    suffix = f" — {detail}" if detail else ""
    print(f"{C.YELLOW}  ↳ SERVICE:{C.RESET} {C.DIM}{name}{suffix}{C.RESET}")


def log_info(text: str):
    print(f"{C.BLUE}ℹ {text}{C.RESET}")


def log_voice(direction: str, text: str):
    arrow = "🎤→" if direction == "in" else "🔊←"
    print(f"{C.CYAN}{arrow}{C.RESET} {text}")
