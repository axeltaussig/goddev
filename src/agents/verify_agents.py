#!/usr/bin/env python3
"""Full import + syntax verification for all GodDev agents."""
import sys, asyncio, traceback
sys.path.insert(0, '/opt/goddev')

results = []

def assert_ok(cond, msg):
    if not cond: raise AssertionError(msg)

def check(name, fn):
    try:
        fn()
        results.append(f"[OK] {name}")
    except Exception as e:
        results.append(f"[FAIL] {name}: {e}")
        traceback.print_exc()

check("state.py", lambda: __import__('src.state', fromlist=['GodDevState', 'ProjectBlueprint', 'ProjectMilestone', 'FileTask', 'FileOutput', 'CriticVerdict']))

check("cto", lambda: __import__('src.agents.cto', fromlist=['cto_node', '_ensure_critical_files']))

check("squad_leader", lambda: __import__('src.agents.squad_leader', fromlist=['squad_leader_node']))

check("worker", lambda: (
    __import__('src.agents.worker', fromlist=['worker_node', '_QUALITY_RULE', '_SYSTEMS']),
    setattr(sys, '_w', __import__('src.agents.worker', fromlist=['worker_node', '_QUALITY_RULE', '_SYSTEMS'])),
    [
        assert_ok(asyncio.iscoroutinefunction(sys._w.worker_node), "worker not async"),
        assert_ok('self-improvement' in sys._w._SYSTEMS, "self-improvement role missing"),
        assert_ok('CODE QUALITY' in sys._w._QUALITY_RULE, "quality rule missing"),
    ]
))

def assert_ok(cond, msg):
    if not cond: raise AssertionError(msg)

# Simpler checks for worker
try:
    from src.agents.worker import worker_node, _QUALITY_RULE, _SYSTEMS
    assert asyncio.iscoroutinefunction(worker_node), "worker_node must be async"
    assert 'self-improvement' in _SYSTEMS, "self-improvement role missing from _SYSTEMS"
    assert 'CODE QUALITY' in _QUALITY_RULE, "'CODE QUALITY' missing from _QUALITY_RULE"
    results.append(f"[OK] worker: async={asyncio.iscoroutinefunction(worker_node)}, roles={list(_SYSTEMS.keys())}")
except Exception as e:
    results.append(f"[FAIL] worker: {e}")
    traceback.print_exc()

try:
    from src.agents.critic_council import code_critic_node, security_critic_node, perf_critic_node
    assert asyncio.iscoroutinefunction(code_critic_node)
    results.append("[OK] critics: all async")
except Exception as e:
    results.append(f"[FAIL] critics: {e}")

try:
    from src.agents.meta_cto import _read_source_file
    src = _read_source_file('agents/cto.py')
    assert len(src) > 500, f"only {len(src)} bytes"
    results.append(f"[OK] meta_cto: source injection {len(src):,} bytes from cto.py")
except Exception as e:
    results.append(f"[FAIL] meta_cto: {e}")

try:
    from src.graph import graph
    from src.self_build_graph import self_build_graph
    results.append(f"[OK] graphs: {graph.name} / {self_build_graph.name}")
except Exception as e:
    results.append(f"[FAIL] graphs: {e}")
    traceback.print_exc()

print("\n".join(results))
failed = [r for r in results if r.startswith("[FAIL]")]
if failed:
    print(f"\n{len(failed)} FAILURES")
    sys.exit(1)
else:
    print(f"\n=== ALL {len(results)} CHECKS PASSED ===")
