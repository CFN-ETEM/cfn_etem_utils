from tkinter import N
import ipyparallel as ipp
from datetime import datetime

expected_time = 20.0

c = ipp.Client(connection_info=f"ipypar/security/ipcontroller-client.json")
with c[:].sync_imports():
    from datetime import datetime

def nonsense_work(lifetime):
    t0 = datetime.now()
    while True:
        for _ in range(10000000):
            continue
        t1 = datetime.now()
        t_passed = (t1-t0).total_seconds()
        if t_passed > lifetime:
            break

c[:].map_sync(nonsense_work, [expected_time]*len(c.ids))


