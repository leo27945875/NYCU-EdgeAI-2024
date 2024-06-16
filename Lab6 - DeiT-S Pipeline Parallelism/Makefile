RANK = ${r}
FILE = ${f}
PORT = ${p}

ifdef nn
  N_NODES = ${nn}
else
  N_NODES = 4
endif

ifdef np
  N_PROCS = ${np}
else
  N_PROCS = 1
endif


all:
	torchrun --nnodes=${N_NODES} --nproc-per-node=${N_PROCS} --node-rank=${RANK} --master-addr=192.168.1.112 --master-port=1222 template.py

pscp:
	@parallel-scp -h ../host.txt ./${FILE} ~/lab6/

ssh:
	ssh 140.113.194.102 -p ${PORT}

prepare:
	make pscp f=template.py
	make pscp f=util.py
	make pscp f=serial_deit.py

showprocs:
	@echo "------------------------------------------------------"
	@ps aux | grep -E "template.py|serial_deit.py" | grep -v grep | awk '{print $2}' 
	@echo "------------------------------------------------------"

test:
	./run.sh ${N_NODES} ${N_PROCS} ${RANK} 192.168.1.112