#!/usr/bin/env python3
"""
Cost-effective multi-role agent. Uses engines by strength:
- Claude Opus ($15/M): architecture, strategy, deep review (rarely, high value)
- GPT-4.1 ($10/M): backend, complex implementation, planning
- DeepSeek Pro ($2/M): general implementation, refactoring
- DeepSeek Chat ($0.27/M): exploration, grunt work, simple edits
- Gemini Flash ($0.15/M): cheapest, fastest for file ops
- Each role runs multiple steps until done
"""
import requests, json, sys, os, subprocess, re
from pathlib import Path

# Load .env for local development (safe: .env is gitignored)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / '.env'
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

WORKSPACE = os.getenv("GODDEV_WORKSPACE", "/opt/goddev")
MAX_BUDGET = 2.00

KEYS = {
    "deepseek": os.getenv("DEEPSEEK_API_KEY", "sk-placeholder-replace-in-env"),
    "openai": os.getenv("OPENAI_API_KEY", "sk-placeholder-replace-in-env"),
    "gemini": os.getenv("GEMINI_API_KEY", "AIza-placeholder-replace-in-env"),
    "claude": os.getenv("ANTHROPIC_API_KEY", "sk-ant-placeholder-replace-in-env"),
}

PRICING = {
    "deepseek-v4-flash": (0.27, 1.10), "gemini-2.5-flash": (0.15, 0.60),
    "deepseek-chat": (0.27, 1.10), "deepseek-v4-pro": (2.00, 8.00),
    "gpt-4.1-mini": (2.00, 8.00), "gpt-4.1": (10.00, 30.00),
    "claude-sonnet-4-6": (8.00, 40.00), "claude-opus-4-7": (15.00, 75.00),
    "gemini-1.5-pro": (3.50, 10.50),
}

ENDPOINTS = {
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    "claude": "https://api.anthropic.com/v1/messages",
}

# Each role has: model chain, cost tier, task type, and how many actions to take
ROLES = [
    ("architect", [
        ("claude-opus-4-7", "claude", 1, "premium"),     # $15/M — strategic design
        ("gemini-1.5-pro", "gemini", 2, "mid"),            # $3.50/M — alternative
        ("gpt-4.1", "openai", 2, "expensive"),            # $10/M — fallback
    ], [
        ("Explore codebase structure", "cheap"),
        ("Read key source files", "cheap"),
        ("Produce architecture plan with specific changes", "premium"),
    ]),
    ("backend", [
        ("gpt-4.1", "openai", 2, "expensive"),            # $10/M — complex backend
        ("deepseek-v4-pro", "deepseek", 3, "mid"),         # $2/M — general backend
        ("deepseek-chat", "deepseek", 5, "cheap"),         # $0.27/M — simple backend
    ], [
        ("Read existing backend code", "cheap"),
        ("Write/update API routes", "mid"),
        ("Write/update models and schemas", "mid"),
        ("Write/update business logic", "cheap"),
        ("Verify backend works", "cheap"),
    ]),
    ("frontend", [
        ("claude-opus-4-7", "claude", 1, "premium"),      # $15/M — UI design
        ("gemini-2.5-flash", "gemini", 3, "cheap"),        # $0.15/M — fast UI work
        ("deepseek-chat", "deepseek", 4, "cheap"),         # $0.27/M — fallback
    ], [
        ("Read existing frontend code", "cheap"),
        ("Write/update UI components", "mid"),
        ("Write/update styles and layouts", "cheap"),
        ("Polish and verify frontend", "cheap"),
    ]),
    ("devops", [
        ("deepseek-chat", "deepseek", 4, "cheap"),         # $0.27/M — configs
        ("gpt-4.1-mini", "openai", 2, "mid"),              # $2/M — deployment logic
    ], [
        ("Read existing configs", "cheap"),
        ("Write/update deployment scripts", "cheap"),
        ("Write/update CI/CD configs", "mid"),
        ("Verify configs work", "cheap"),
    ]),
    ("qa", [
        ("gpt-4.1", "openai", 2, "expensive"),            # $10/M — test strategy
        ("deepseek-chat", "deepseek", 5, "cheap"),         # $0.27/M — write tests
        ("deepseek-v4-pro", "deepseek", 3, "mid"),         # $2/M — complex tests
    ], [
        ("Read source code to identify test areas", "cheap"),
        ("Write unit tests", "cheap"),
        ("Write integration tests", "mid"),
        ("Run tests and report results", "cheap"),
    ]),
    ("critic", [
        ("claude-opus-4-7", "claude", 1, "premium"),      # $15/M — deep review
        ("gpt-4.1", "openai", 2, "expensive"),            # $10/M — review
    ], [
        ("Read written code files", "cheap"),
        ("Produce code review with improvement suggestions", "premium"),
    ]),
    ("cto", [
        ("claude-opus-4-7", "claude", 1, "premium"),      # $15/M — strategic
        ("gemini-1.5-pro", "gemini", 2, "mid"),            # $3.50/M — strategic alt
    ], [
        ("Read architecture and key files", "cheap"),
        ("Evaluate technical decisions", "premium"),
        ("Provide strategic recommendations", "premium"),
    ]),
    ("integrator", [
        ("gemini-2.5-flash", "gemini", 3, "cheap"),        # $0.15/M — fast integration
        ("gpt-4.1-mini", "openai", 3, "mid"),              # $2/M — integration logic
    ], [
        ("Verify all pieces work together", "cheap"),
        ("Run the application", "cheap"),
        ("Report integration status", "mid"),
    ]),
    ("squad_lead", [
        ("gpt-4.1", "openai", 1, "expensive"),            # $10/M — final summary
        ("deepseek-chat", "deepseek", 3, "cheap"),         # $0.27/M — simple summary
    ], [
        ("Summarize all work done", "mid"),
        ("List remaining tasks and next steps", "cheap"),
    ]),
]

total_cost = 0.0
session_start = 0.0

# Cost per call by tier
TIER_COST = {"cheap": 0.0003, "mid": 0.002, "expensive": 0.008, "premium": 0.015}

def call_model(model_name, provider, messages, max_tok=5000):
    global total_cost
    key = KEYS.get(provider)
    if not key: return None, 0.0, "No key"
    url = ENDPOINTS.get(provider, "").replace("{model}", model_name)
    try:
        if provider == "claude":
            cm = [m for m in messages if m["role"] in ("user","assistant")] or [{"role":"user","content":"Go"}]
            r = requests.post(url, headers={"x-api-key":key,"anthropic-version":"2023-06-01","Content-Type":"application/json"},
                             json={"model":model_name,"max_tokens":max_tok,"messages":cm[-10:]}, timeout=180)
            d = r.json()
            if r.status_code != 200: return None, 0.0, str(d.get("error",{}).get("message",""))
            content = d["content"][0]["text"]
            inp = d.get("usage",{}).get("input_tokens",0)
            out = d.get("usage",{}).get("output_tokens",0)
        elif provider == "gemini":
            gm = [{"parts":[{"text":m["content"]}]} for m in messages if m["role"]=="user"]
            r = requests.post(url+"?key="+key, json={"contents":gm[-5:],"generationConfig":{"maxOutputTokens":max_tok}}, timeout=180)
            d = r.json()
            if r.status_code != 200: return None, 0.0, str(d.get("error",{}).get("message",""))
            content = d["candidates"][0]["content"]["parts"][0]["text"]
            inp = d.get("usageMetadata",{}).get("promptTokenCount",0)
            out = d.get("usageMetadata",{}).get("candidatesTokenCount",0)
        else:
            r = requests.post(url, headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
                             json={"model":model_name,"messages":messages[-15:],"temperature":0.3,"max_tokens":max_tok}, timeout=180)
            d = r.json()
            if r.status_code != 200: return None, 0.0, d.get("error",{}).get("message",str(d))
            content = d["choices"][0]["message"]["content"]
            inp = d.get("usage",{}).get("prompt_tokens",0)
            out = d.get("usage",{}).get("completion_tokens",0)
        ip, op = PRICING.get(model_name, (2.0,8.0))
        cost = (inp/1e6)*ip + (out/1e6)*op
        total_cost += cost
        return content, cost, None
    except Exception as e:
        return None, 0.0, str(type(e).__name__)

def run_cmd(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=WORKSPACE, timeout=60)
        return (r.stdout or "")[:5000] + (r.stderr or "")[:1000]
    except: return "Error"

def read_file(path):
    full = path if path.startswith("/") else os.path.join(WORKSPACE, path)
    if os.path.exists(full):
        try: return open(full).read()[:5000]
        except: return "Error"
    return "Not found"

def write_file(path, content):
    full = path if path.startswith("/") else os.path.join(WORKSPACE, path)
    try:
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, 'w').write(content)
        return f"Written {len(content)}b"
    except Exception as e:
        return f"Error: {e}"

def parse_action(reply):
    lines = reply.strip().split("\n")
    for i, line in enumerate(lines):
        l = line.strip()
        if len(l) >= 2 and l[0] in ("R","F","W","D") and l[1] == ":":
            rest = l[2:].strip()
            content = "\n".join(lines[i+1:]).strip()
            content = re.sub(r'^```\w*\n?','',content)
            content = re.sub(r'\n?```$','',content)
            return l[0], rest, content
        if len(l) >= 3 and l[0] in ("R","F","W","D") and l[1:3] == ": ":
            rest = l[3:].strip()
            content = "\n".join(lines[i+1:]).strip()
            content = re.sub(r'^```\w*\n?','',content)
            content = re.sub(r'\n?```$','',content)
            return l[0], rest, content
    return None, "", reply

print()
print("=" * 65)
print("  Multi-Role Agent — Cost-Effective Engine Allocation")
print("  Premium models for strategy | Cheap models for grunt work")
print(f"  Budget: ${MAX_BUDGET:.2f}/run")
print("  /cost  /budget  /exit")
print("=" * 65)
print()

while True:
    try:
        if total_cost - session_start >= MAX_BUDGET:
            print(f"\n  Budget used (${total_cost-session_start:.2f}). New run.\n")
            session_start = total_cost
            continue

        inp = input("\nYou: ").strip()
        if not inp: continue
        if inp.lower() in ("exit","quit","q"): break
        if inp == "/cost":
            print(f"  Run: ${total_cost-session_start:.4f}  Total: ${total_cost:.4f}")
            continue
        if inp == "/budget":
            print(f"  Remaining: ${MAX_BUDGET-(total_cost-session_start):.4f}")
            continue

        session_start = total_cost
        pipeline_log = f"TASK: {inp}\nWORKSPACE: {WORKSPACE}\n"

        for role_name, model_chain, subtasks in ROLES:
            remaining = MAX_BUDGET - (total_cost - session_start)
            if remaining <= 0.05:
                print(f"\n  Budget low (${remaining:.4f}). Stopping pipeline.\n")
                break

            print(f"\n  ╔══ {role_name.upper()} ═══")
            
            # Show which models will be used
            model_info = []
            for m, p, r, t in model_chain:
                iprice = PRICING.get(m, (0,0))[0]
                label = "★" if t == "premium" else ("▲" if t == "expensive" else ("●" if t == "mid" else "·"))
                model_info.append(f"{label} {m} (${iprice:.0f}/M x{r})")
            print(f"  ║  {' | '.join(model_info)}")

            # Run each subtask for this role
            for st_idx, (st_desc, st_tier) in enumerate(subtasks):
                if total_cost - session_start >= MAX_BUDGET: break
                remaining = MAX_BUDGET - (total_cost - session_start)
                if remaining <= 0.02: break

                # Find the cheapest model chain that can handle this subtask's tier
                chosen_model = None
                chosen_provider = None
                max_attempts = 1
                # Walk the model chain, pick first that matches tier or better
                for m, p, r, t in model_chain:
                    if t == st_tier or t == "premium" or (t == "expensive" and st_tier != "premium"):
                        chosen_model, chosen_provider, max_attempts = m, p, r
                        break
                if not chosen_model:
                    chosen_model, chosen_provider, max_attempts = model_chain[0][:3]

                iprice = PRICING.get(chosen_model, (0,0))[0]
                print(f"  ║  [{st_idx+1}/{len(subtasks)}] {st_desc}")
                print(f"  ║    → {chosen_model} (${iprice:.2f}/M)", end="", flush=True)

                est = TIER_COST.get(st_tier, 0.001)
                context = (
                    f"You are {role_name}. Task: {inp}\n"
                    f"Subtasks for this role: {', '.join(s[0] for s in subtasks)}\n"
                    f"Current subtask: {st_desc}\n"
                    f"Budget: ${total_cost-session_start:.4f} / ${MAX_BUDGET:.2f}\n"
                    f"Context so far:\n{pipeline_log[-2000:]}\n\n"
                    f"Output ONE action:\n"
                    f"  W: <path>  + code content  → write code\n"
                    f"  R: <cmd>                     → run command\n"
                    f"  F: <path>                    → read file\n"
                    f"  D: <done>                    → or D: <summary>\n\n"
                    f"Prefer to write code (W:) or run commands (R:).\n"
                    f"Actually modify files. Do work."
                )

                reply, cost, err = call_model(
                    chosen_model, chosen_provider,
                    [{"role":"user","content":context}], 5000
                )
                if not reply:
                    print(f" ✗ {err}", flush=True)
                    continue

                print(f" ${cost:.6f}", flush=True)
                action_type, action_val, content = parse_action(reply)

                if action_type == "R":
                    result = run_cmd(action_val)
                    print(f"  ║    $ {action_val[:60]}")
                    print(f"  ║    {result[:180].replace(chr(10),' ')}", flush=True)
                    pipeline_log += f"\n[{role_name}] R: {action_val}\n{result[:1000]}\n"
                elif action_type == "F":
                    fcontent = read_file(action_val)
                    print(f"  ║    Read {action_val} [{len(fcontent)} chars]", flush=True)
                    pipeline_log += f"\n[{role_name}] F: {action_val}\n"
                elif action_type == "W":
                    wresult = write_file(action_val, content)
                    print(f"  ║    {wresult}", flush=True)
                    pipeline_log += f"\n[{role_name}] W: {action_val}\n"
                elif action_type == "D":
                    print(f"  ║    {action_val[:100]}", flush=True)
                    pipeline_log += f"\n[{role_name}] D: {action_val}\n"
                else:
                    print(f"  ║    {reply[:150].replace(chr(10),' ')}", flush=True)
                    pipeline_log += f"\n[{role_name}] {reply[:500]}\n"

                used_now = total_cost - session_start
                bars = int(20 * min(used_now / MAX_BUDGET, 1.0))
                print(f"  ║    ${used_now:.4f} / ${MAX_BUDGET:.2f} [{'█'*bars}{'░'*(20-bars)}]", flush=True)

                if action_type == "D":
                    break

            role_cost = total_cost - session_start
            role_in_session = total_cost - session_start
            print(f"  ╚══ {role_name} done — ${role_cost:.4f} total\n")

        final = total_cost - session_start
        bars = int(20 * min(final / MAX_BUDGET, 1.0))
        print(f"{'='*50}")
        print(f"  Pipeline complete | ${final:.4f} | {'█'*bars}{'░'*(20-bars)}")
        if final < 1.0:
            print(f"  (${MAX_BUDGET - final:.2f} remaining — add more tasks)")
        print(f"{'='*50}")

    except KeyboardInterrupt:
        print(f"\nTotal: ${total_cost:.4f}")
        break
    except Exception as e:
        print(f"\nError: {e}")
        import traceback; traceback.print_exc()
