import torch
import timeit

print(f'Model: {torch.cuda.get_device_name(0)}')

#--------------------------------- Use CPU -----------------------------------------
def use_cpu():
    device = torch.device('cpu')


    dim=8000
    tensor_a = torch.randn(dim,dim, device=device)
    tensor_b = torch.randn(dim,dim, device=device) 

    torch.matmul(tensor_a, tensor_b)

#-----------------------------------------------------------------------------------

#--------------------------------- Use GPU -----------------------------------------
def use_gpu():
    device = torch.device('cuda')

    dim=8000
    tensor_a = torch.randn(dim,dim, device=device)
    tensor_b = torch.randn(dim,dim, device=device) 

    torch.matmul(tensor_a, tensor_b)
#-----------------------------------------------------------------------------------

time_1 = timeit.timeit(use_cpu, number=10) / 10
print(f'Avg time for CPU: {time_1}')
time_2 = timeit.timeit(use_gpu, number=10) / 10
print(f'Avg time for GPU: {time_2}')
speed_up = round(time_1 / time_2, 2)
print(f'GPU is {speed_up} times faster than the CPU')