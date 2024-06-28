import multiprocessing as mp
import ctypes
import time
import os
import wandb
from gops.nodes.node import Node


class MetricNode(Node):
    @staticmethod
    def node_create_shared_objects(node_class: str, num: int, ns_config: dict):
        objects = Node.node_create_shared_objects(node_class, num, ns_config)

        assert num == 1, "MetricNode: There must be only one metric node."
        objects[0].update({
            "lock": mp.Lock(),
            "tick": mp.RawValue(ctypes.c_int64, 0)
        })
        return objects

    def get_run_label(self):
        env_config = self.ns_config.get("env", {})

        env_name = self.all_args.get("env_id", "")
        env_params_str = ", ".join(k + "=" + v.__repr__() for k, v in env_config.get("params", {}).items())

        algo_name = self.all_args.get("algorithm", "UnknownAlgorithm")
        
        env_node_num = self.all_args.get("env_node_num", "")
        opt_node_num = self.all_args.get("opt_node_num", "")
        if env_node_num != "" and opt_node_num != "":
            node_info = f"envx{env_node_num}_optx{opt_node_num}"
        else:
            node_info = ""
        project_sup = self.all_args.get("wandb_project_sup", "")
        return {
            "project": "[GOPS] {} {}".format(env_name, project_sup),
            "name": "{} Distr {} {}".format(algo_name, node_info, time.strftime("%H:%M %m-%d %Y"))
        }

    def object_to_string_dict(self, obj):
        if isinstance(obj, list):
            return [self.object_to_string_dict(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self.object_to_string_dict(v) for k, v in obj.items()}
        else:
            return obj.__repr__()

    def run(self):
        # shared objs
        shared_lock = self.objects["lock"]
        shared_tick = self.objects["tick"]
        # update to data & ticks per second
        utd_num_updates = 0
        utd_last_ticks = 0
        utd_last_time = time.time()

        utd_log_interval = self.config.get("utd_log_interval", 1.0)

        # initialize
        wandb.init(**self.get_run_label(), mode=self.all_args["wandb_mode"], config={
            "config": self.ns_config,
            "objects": self.object_to_string_dict(self.global_objects)
        })

        # event loop
        while True:
            metric = self.recv()
            # get ticks
            shared_lock.acquire()
            tick = shared_tick.value
            shared_lock.release()

            # update to data
            if "update" in metric:
                utd_num_updates += metric["update"]
                del metric["update"]

                current_time = time.time()
                if (current_time - utd_last_time >= utd_log_interval) and (tick > utd_last_ticks):
                    wandb.log({
                        "update_per_data": utd_num_updates / (tick - utd_last_ticks),
                        "ticks_per_sec": (tick - utd_last_ticks) / (current_time - utd_last_time)
                    }, step=tick)

                    utd_num_updates = 0
                    utd_last_ticks = tick
                    utd_last_time = current_time

            # save model
            if "save_model" in metric:
                wandb.save(metric["save_filename"])
                continue
    
            # log metric
            wandb.log(metric, step=tick)