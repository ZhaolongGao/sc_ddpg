Train the network
```bash
mpiexec -np 4 python -u train_agent.py
```

After trained, use the trained weight for test
```bash
python test_agent.py
```