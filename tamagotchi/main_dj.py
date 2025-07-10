from schemas.schema_v6 import * # use schema_v2 for wind sensing exps.; v3 for app_wind + visual db 
from multiprocessing import Process, Pipe
import sys

if __name__ == '__main__':
    num_of_processes = int(sys.argv[1])
    processes = []
    kw={'reserve_jobs':True} # True to ensure no duplicate jobs are queued. Not currently working. Only run with ONE subprocess
    if num_of_processes == 1:
        TrainingResult.populate(**kw)
    else:
        for i in range(num_of_processes):
            processes.append(Process(target=TrainingResult.populate, kwargs=kw))

        for p in processes:
            p.start()

        for p in processes:
            p.join()