import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import numpy as np 

"""Non-blocking point-to-point communication."""

def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait() # wait here for communication to finish
    print('Rank ', rank, ' has data ', tensor[0])

def test_run(rank, size):
    """ Simple point-to-point communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])

def ring_reduce(rank, size):
    print('Starting process {}'.format(rank))
    dist.barrier()

    grad_vector_size = 8

    # print('For GPU {}, Send Pos {}, Recv Pos {}'.format(rank,send_pos,recv_pos))

    torch.random.manual_seed(rank)
    data = torch.zeros(grad_vector_size) # to check if reduce happens
    data = torch.ones(grad_vector_size) # to check if gather stage happens
    chunks = list(torch.chunk(data, size))
    # chunks[send_pos] = torch.rand(len(chunks[send_pos]))
    
    data = torch.cat(chunks)
    print('GPU {} has tensor (Overall view) {}'.format(rank, data))   

    iterations = size - 1

    dist.barrier()
    recv_pos = ((rank-1)+size)%size 
    send_pos = (rank)%size
    for i in range(iterations):

        sent_chunk = chunks[send_pos]
        
        recv_buff = torch.zeros_like(chunks[recv_pos])
        send_buff = torch.zeros_like(chunks[send_pos])

        send_req = dist.isend(sent_chunk,(rank+1)%size)
        dist.recv(recv_buff,((rank-1)+size)%size)

        chunks[recv_pos] += recv_buff[:]

        data = torch.cat(chunks)


        # print('Iteration: {}, GPU {} has tensor {}'.format(i, rank, data))   
        
        recv_pos = ((recv_pos - 1)+size)%size
        send_pos = ((send_pos - 1)+size)%size

        send_req.wait()

        dist.barrier()


    print('Reduced, GPU {} has tensor {}'.format(rank, data))   

    dist.barrier()

    send_pos = (recv_pos+1)%size
    recv_pos = ((send_pos - 1)+size)%size

    for i in range(iterations):
        sent_chunk = chunks[send_pos]
        
        recv_buff = torch.zeros_like(chunks[recv_pos])
        send_buff = torch.zeros_like(chunks[send_pos])

        send_req = dist.isend(sent_chunk,(rank+1)%size)
        dist.recv(recv_buff,((rank-1)+size)%size)

        chunks[recv_pos] = recv_buff[:]

        data = torch.cat(chunks)

        # print('Iteration: {}, GPU {} has tensor {}'.format(i, rank, data))   
        
        recv_pos = ((recv_pos - 1)+size)%size
        send_pos = ((send_pos - 1)+size)%size

        send_req.wait()

        dist.barrier()


    print('Gathered, GPU {} has tensor {}'.format(rank, data))  

    # dist.all_reduce(data, op=dist.ReduceOp.SUM, group=group)
    # print('After all-gather, Rank {} has tensor {}'.format(rank, data))   

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 8
    processes = []
    for rank in range(size):
        # p = Process(target=init_process, args=(rank, size, run))
        p = Process(target=init_process, args=(rank, size, ring_reduce))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()