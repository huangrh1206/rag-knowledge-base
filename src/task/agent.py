from __future__ import annotations
from .models import TaskResult
from .runtime import TaskRuntime
from .tools import TaskToolRegistry, TaskToolExecutor

class CalculatorTool:
    name, description = "calculator", "Evaluate arithmetic expressions"
    def invoke(self, arguments): return eval(str(arguments.get("expression", "0")), {"__builtins__": {}}, {})

class FileTool:
    name, description = "file", "Read or write a file"
    def invoke(self, arguments):
        from pathlib import Path
        path = Path(arguments["path"])
        if arguments.get("operation", "read") == "write": path.write_text(str(arguments.get("content", "")), encoding="utf-8"); return True
        return path.read_text(encoding="utf-8")

class HttpTool:
    name, description = "http", "HTTP adapter"
    def invoke(self, arguments): raise RuntimeError("HTTP tool requires an injected implementation")

class GeneralTaskAgent:
    def __init__(self, runtime: TaskRuntime | None = None, registry: TaskToolRegistry | None = None):
        self.registry = registry or TaskToolRegistry([CalculatorTool(), FileTool(), HttpTool()])
        self.runtime = runtime or TaskRuntime(scheduler=__import__("src.task.scheduler", fromlist=["SuperstepScheduler"]).SuperstepScheduler(TaskToolExecutor(self.registry)))
    def run(self, goal: str) -> TaskResult:
        return self.runtime.start(goal.__hash__().__str__(), goal)

