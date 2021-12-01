import datetime
import os
import pprint

from hierarchical_decision.hier_decision import hierarchical_decision
from hierarchical_decision.hier_decision import load_json

"""
This need to be tested
"""

def main():
    iteration = 299000
    max_steps = 300
    exp_dir = "./results/HorizonCrossing-v0/experiment-2021-11-25-19-42-33"
    args = load_json(exp_dir)
    # change mode to selecting
    args.mode = "selecting"

    trace = {
        "exp_dir": exp_dir,
        "env_id": args.env_id,
        "training_task": args.env_kwargs_training_task,
        "commit_id": args.commit_id,
        "git_branch": args.branch_name,
    }
    pprint.pprint(trace)

    # make logdir
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(os.path.abspath(exp_dir), "select", time_now)
    os.makedirs(logdir, exist_ok=True)

    idc = hierarchical_decision(args, exp_dir, iteration, logdir)
    for i in range(max_steps):
        done = 0
        while not done:
            done = idc.step()
        idc.reset()

if __name__ == "__main__":
    main()
