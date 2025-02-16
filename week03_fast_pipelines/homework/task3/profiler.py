import json
import time
import torch
from collections import defaultdict

import pandas as pd
    
    
class Schedule:
    def __init__(self, wait: int, warmup: int, active: int, repeat: int):
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.current_step = 0
        self.repeat_step = 1
    
    def step(self):
        self.current_step += 1
    
    def total_steps(self):
        return (self.wait + self.warmup + self.active) * self.repeat

    def get_phase(self):
        if self.current_step < self.wait:
            return "wait"
        elif self.current_step < self.wait + self.warmup:
            return "warmup"
        elif self.current_step < self.wait + self.warmup + self.active:
            return "active"
        elif self.repeat_step < self.repeat:
            self.repeat_step += 1
            self.current_step = 0
            return self.get_phase()
        else:
            return "done"
            

class Profile:
    def __init__(self, model, name="model", schedule: Schedule | None = None):
        self.name_map = self._build_name_map(model, name)
        self.events = []
        self.schedule = schedule
        
        for module in model.children():
            self._attach_hooks(module)
    
    def _attach_hooks(self, module):
        module.register_forward_pre_hook(self._forward_pre_hook)
        module.register_forward_hook(self._forward_post_hook)
        module.register_full_backward_pre_hook(self._backward_pre_hook)
        module.register_full_backward_hook(self._backward_post_hook)
        for child in module.children():
            self._attach_hooks(child)
        
    def _build_name_map(self, model, name="model"):
        name_map = {}
        for full_name, module in model.named_modules():
            if full_name == "":
                full_name = name

            if self._is_leaf(module):
                name_map[module] = module.__class__.__name__
            else:
                name_map[module] = f"{full_name}: {module.__class__.__name__}"

        return name_map

    def _is_leaf(self, module):
        return len(list(module.children())) == 0

    def _forward_pre_hook(self, module, inputs):
        step = self.schedule.get_phase()
        if step in ["wait", "done"]:
            return
        if step in ["warmup", "active"]:
            event = {
                "module": f"{self.name_map[module]}",
                "function": "forward",
                "type": "start",
                "ts": torch.cuda.Event(enable_timing=True),
            }
            event["ts"].record()
        if step == "active":
            self.events.append(event)
        

    def _forward_post_hook(self, module, inputs, outputs):
        step = self.schedule.get_phase()
        if step in ["wait", "done"]:
            return
        if step in ["warmup", "active"]:
            event = {
                "module": f"{self.name_map[module]}",
                "function": "forward",
                "type": "end",
                "ts": torch.cuda.Event(enable_timing=True),
            }
            event["ts"].record()
        if step == "active":
            self.events.append(event)


    def _backward_pre_hook(self, module, grad_output):
        step = self.schedule.get_phase()
        if step in ["wait", "done"]:
            return
        if step in ["warmup", "active"]:
            event = {
                "module": f"{self.name_map[module]}",
                "function": "backward",
                "type": "start",
                "ts": torch.cuda.Event(enable_timing=True),
            }
            event["ts"].record()
        if step == "active":
            self.events.append(event)

    def _backward_post_hook(self, module, grad_input, grad_output):
        step = self.schedule.get_phase()
        if step in ["wait", "done"]:
            return
        if step in ["warmup", "active"]:
            event = {
                "module": f"{self.name_map[module]}",
                "function": "backward",
                "type": "end",
                "ts": torch.cuda.Event(enable_timing=True),
            }
            event["ts"].record()
        if step == "active":
            self.events.append(event)

    def __enter__(self):
        event = {
            "module": f"Profile",
            "function": "__enter__",
            "type": "base_ts",
            "ts": torch.cuda.Event(enable_timing=True),
        }
        event["ts"].record()
        self.events.append(event)
        return self
 
    def __exit__(self, type, value, traceback):
        self._process_events()


    def step(self):
        torch.cuda.synchronize()
        self.schedule.step()
    
    def _process_events(self):
        stack = []
        events = []
        start_ts = self.events[0]["ts"]
        for event in self.events[1:]:
            if event["type"] == "start":
                stack.append(event)
            elif (
                event["type"] == "end"
                and stack[-1]["module"] == event["module"]
                and stack[-1]["function"] == event["function"]
                and stack[-1]["type"] == "start"
            ):
                start_event = stack.pop()
                dur = start_event["ts"].elapsed_time(event["ts"])
                ts = start_ts.elapsed_time(start_event["ts"])
                events.append({
                    "duration (ms)": dur,
                    "module": start_event["module"],
                    "function": start_event["function"],
                    "ts (ms)": ts,
                })
        self.events = events

    def summary(self):
        summary = pd.DataFrame(
            pd.DataFrame(self.events).groupby(["function", "module"])["duration (ms)"].mean()
        )
        return summary

    def to_perfetto(self, path="trace.json"):
        events = []
        for event in self.events:
            events.append({
                "name": event["module"],
                "ph": "X",
                "ts": event["ts (ms)"],
                "dur": event["duration (ms)"],
                "pid": 0,
                "tid": 0,
            })
        json.dump(events, open(path, "w"))
