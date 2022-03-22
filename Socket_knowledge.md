# RPC
## 1. RPC call （远程调用）
主要就是rpc_sync同步远程调用、rpc_async异步远程调用、remote异步远程调用。下方为例子
首先编写一个master.py文件：
>```python  
>import os  
>import torch  
>import torch.distributed.rpc as rpc  
>
>os.environ\['MASTER_ADDR'] = '172.16.138.65'  
>os.environ\['MASTER_PORT'] = '7030'  
>
>def syszuxAdd(t1, t2):  
>    print("syszuxAdd call in master")  
>    return torch.add(t1, t2)  
>    
>rpc.init_rpc("master", rank=0, world_size=2)  
>
>rpc.shutdown()  

再编写一个worker.py文件
>```python  
>rpc.init_rpc("worker", rank=1, world_size=2)  
>
>gemfield1 = rpc.rpc_async("master", torch.add, args=(torch.ones(2), 3))  
>gemfield2 = rpc.rpc_async("master", min, args=(1, 2))  
>gemfield3 = rpc.rpc_async("master", syszuxAdd, args=(torch.ones(2), torch.Tensor(\[7029])))  
>
>result = gemfield1.wait() + gemfield2.wait()  
>print("gemfield: ", result)  
>print("gemfield3: ", gemfield3.wait())  
>
>rpc.shutdown()  

看似上面的三个函数在worker处运行，其实执行在了master上 
**RPC call实现了执行远端机器上的进程中的函数、可以把输入传递过去、可以把输出的值拿回来。**

## 2. RREF   （远端引用，跨机器的变量引用）
**对结果的引用--调用者并不想拿到具体的返回值**
**在多机器交互调用中十分重要**
首先要再次强调的是，在使用RRef之前，我们必须要先初始化RPC框架。再然后，我们只要谨记RRef的特质,\
在PyTorch的常见场景下，我们甚至可以简化为：1个RRef就是对1个Tensor的引用，可以是本地tensor的引用，\
也可以是对远端机器上的tensor的引用。比如：
>```python   
>#创建一个远端引用，引用远端机器上的worker1进程中的1个tensor  
>rref = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))  
>  
>#使用remote本意就是没想把具体的值拿回来，但如果你就是要拿回来，可以copy回来  
>x = rref.to_here()  
>  
>#创建一个本地引用  
>rref = RRef(torch.zeros(2, 2))  
**rref是在哪个机器上创建的，就去那个机器上执行method函数，并将结果拿回到当前的机器上。\
这是不是在为拆分模型的参数到不同的机器上铺路？**  
下方为例子

## 3. distributed Autograd 和 distributed.optim
为了适用RPC的调用，PyTorch的Autograd也添加了torch.distributed.autograd模块，\
在模型的训练过程中，我们需要创建distributed autograd context。在此上下文中，前向\
阶段参与RPC调用的Tensor信息会被记录下来，而反向传播的时候我们利用这些信息再通过RPC进行梯度的传递。\

distributed.optim模块中的DistributedOptimizer负责这种情况下的参数优化，它面对的不再是传统的\
parameter tensor，而是封装之后的RRef，借助_remote_method语义，运算会发生在RRef所在的机器上。

## 4. 构造一个基于RPC的网络
在网络的构造阶段，我们可使用remote() rpc call 来将网络的不同部分构造在不同的机器上，并且返回RRef\
到当前的机器上；至此，在前向和反向的体系中，梯度计算和优化器分别使用的是distributed.autograd和\
distributed.optim，它俩将不再面对传统的Parameter tensor，而是封装了Parameter Tensor的RRef.  

在前向阶段，记住我们已经面对的是RRef了。我们可以使用上述提到的_remote_method语义，将运算放在\
构造时候所对应的机器上；也就是说，RRef在哪个机器上，前向运算就在哪个机器上；在反向阶段，我们在\
distributed autograd context中使用distributed.optim来处理RRef即可。
