from queue import Queue
import concurrent.futures
import time
import copy
from evaluation.auto_test import *


def task_done_callback(future, docker_instance, free_dockers):
    free_dockers.put(docker_instance)


def parallel_worker(class_, config, parallel, tasks):
    free_dockers = Queue()
    for idx in range(parallel):
        if config.docker:
            instance = Docker_Instance(config, idx)
        else:
            instance = Instance(config, idx)
        free_dockers.put(instance)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        while tasks or futures:
            # Submit new tasks if there are free instances
            while not free_dockers.empty() and tasks:
                instance = free_dockers.get()
                task = tasks.pop(0)

                config_copy = copy.deepcopy(config)
                auto_class = class_(config_copy)

                future = executor.submit(auto_class.run_task, task, instance)
                future.add_done_callback(lambda fut, di=instance: task_done_callback(fut, di, free_dockers))
                futures.append(future)
            
            # Wait a bit if no tasks can be submitted
            if free_dockers.empty() and tasks:
                time.sleep(0.5)
                continue
            
            # Check for completed futures
            if futures:
                completed = [f for f in futures if f.done()]
                for f in completed:
                    futures.remove(f)
                    try:
                        f.result()  # Raise any exceptions that occurred
                    except Exception as e:
                        print(f"[Parallel] Task failed with error: {e}")
                
                if not tasks and not completed:
                    time.sleep(0.1)  # Small delay when waiting for remaining tasks
