import torch
from torch.utils.tensorboard import SummaryWriter
from network import Network_DW  # 确保 Network 类已经从正确的文件导入

# 创建网络实例
net = Network_DW(nclasses=1)

# 创建一个输入张量
x = torch.randn(1, 3, 200, 400)  # 输入尺寸可以根据你的网络要求调整
y = torch.randn(1, 3, 200, 400)
m1 = torch.randn(1, 3, 200, 400)
m2 = torch.randn(1, 3, 200, 400)
m_obj = torch.randn(1, 3, 200, 400)

# 使用tensorboard记录
writer = SummaryWriter(log_dir='runs/network_dw')
writer.add_graph(net, (x, y, m1, m2, m_obj))
writer.close()

print("使用以下命令启动TensorBoard： tensorboard --logdir=runs")

# 然后在命令行运行上述命令以启动TensorBoard，并在浏览器中查看可视化结果
