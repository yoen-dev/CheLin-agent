# tests/test_rate_limiter.py
"""
测试限流器（对应硬约束 1）。
包含：
  1. 基本行为：窗口内第二条应该被拒绝。
  2. 滑动窗口语义：等窗口过期后应该恢复放行（用极短窗口加速测试）。
  3. 并发测试：多线程同时抢同一个客户的发送名额，
     最终"真正被放行"的次数必须恰好等于上限，一次都不能多。
     （这里用线程而不是多进程，是因为限流器状态本来就设计为
     单进程内共享——这一点在 README 里要说明白，不是回避并发测试，
     而是这个组件的设计边界就是单进程内共享内存，天然不涉及
     跨进程/跨机器竞争，所以线程级并发测试就是这里最真实的攻击方式；
     题目一"任务认领"那种跨数据库连接的并发测试，
     和这里限流器的并发测试，考察的是完全不同的两类并发问题。）
"""

import sys, os, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rate_limiter import SlidingWindowRateLimiter


def test_basic_block_second_message():
    rl = SlidingWindowRateLimiter(window_seconds=60, max_messages=1)
    assert rl.allow("c1") is True
    assert rl.allow("c1") is False
    assert rl.allow("c2") is True, "不同客户互不影响"
    print("PASS: 同一窗口内第二条消息被拒绝，不同客户互相独立")


def test_sliding_window_recovers_after_expiry():
    rl = SlidingWindowRateLimiter(window_seconds=1, max_messages=1)  # 用1秒窗口加速测试
    assert rl.allow("c1") is True
    assert rl.allow("c1") is False
    time.sleep(1.1)
    assert rl.allow("c1") is True, "窗口过期后应该恢复放行"
    print("PASS: 滑动窗口过期后正确恢复放行")


def test_concurrent_requests_never_exceed_limit():
    rl = SlidingWindowRateLimiter(window_seconds=60, max_messages=1)
    results = []
    lock = threading.Lock()

    def worker():
        allowed = rl.allow("c-concurrent")
        with lock:
            results.append(allowed)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    allowed_count = sum(1 for r in results if r)
    assert allowed_count == 1, f"50个并发请求应该只有1个被放行，实际放行了{allowed_count}个"
    print(f"PASS: 50 个并发请求抢同一个客户的发送名额，只有 {allowed_count} 个被放行（预期 1 个）")


if __name__ == "__main__":
    test_basic_block_second_message()
    test_sliding_window_recovers_after_expiry()
    test_concurrent_requests_never_exceed_limit()
    print("\n全部限流器测试通过 ✅")
