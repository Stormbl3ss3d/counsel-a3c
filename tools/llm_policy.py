import re, time, requests
from prompt_format import SYSTEM, option_list, build_state_text

class VLLMPolicy:
    """One instance per worker process. get_action(env, obs, mask) -> env action index."""
    def __init__(self, base="http://localhost:9000", model="bc-llama1b", timeout=10.0, system=SYSTEM):
        self.system = system
        self.url, self.model, self.timeout = f"{base}/v1/chat/completions", model, timeout
        self.session = None                      # created lazily: never share across fork/spawn
        self.lat, self.invalid, self.errors = [], 0, 0

    def get_action(self, env, obs, mask):
        if env.act_comp == -1: return 0
        options = option_list(env, mask)
        if len(options) == 1: return 0           # only "no change" is legal
        if self.session is None: self.session = requests.Session()   # keep-alive
        body = {"model": self.model, "temperature": 0, "max_tokens": 3,
                "messages": [{"role": "system", "content": self.system},
                             {"role": "user", "content": build_state_text(env, obs, options)}]}
        t0 = time.perf_counter()
        try:
            txt = self.session.post(self.url, json=body, timeout=self.timeout).json()["choices"][0]["message"]["content"]
        except Exception:
            self.errors += 1; return 0
        finally:
            self.lat.append(time.perf_counter() - t0)
        m = re.search(r"\d+", txt)
        k = int(m.group()) if m else -1
        if 0 <= k < len(options): return options[k]
        self.invalid += 1; return 0
