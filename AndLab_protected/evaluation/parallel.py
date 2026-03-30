import concurrent.futures
import copy
import time
from queue import Queue

from evaluation.auto_test import *
from utils_mobile.timing_debug import log_timing, timing_enabled


def task_done_callback(future, docker_instance, free_dockers):
    free_dockers.put(docker_instance)
    if timing_enabled():
        exc = future.exception()
        log_timing(
            "Parallel",
            "task_released",
            instance=docker_instance.idx,
            had_exception=exc is not None,
        )


def parallel_worker(class_, config, parallel, tasks):
    """Run ``run_task`` concurrently with ``parallel`` emulator instances (thread pool)."""
    work = list(tasks)
    free_dockers = Queue()
    for idx in range(parallel):
        if config.docker:
            instance = Docker_Instance(config, idx)
        else:
            instance = Instance(config, idx)
        free_dockers.put(instance)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        while work:
            if free_dockers.empty():
                time.sleep(0.5)
                continue

            instance = free_dockers.get()
            task = work.pop(0)

            config_copy = copy.deepcopy(config)
            auto_class = class_(config_copy)
            if timing_enabled():
                log_timing(
                    "Parallel",
                    "task_submitted",
                    task_id=task["task_id"],
                    instance=instance.idx,
                    remaining=len(work),
                    agent_id=id(task["agent"]),
                )

            future = executor.submit(auto_class.run_task, task, instance)
            future.add_done_callback(lambda fut, di=instance: task_done_callback(fut, di, free_dockers))
