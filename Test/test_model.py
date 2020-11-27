from Model import Model
from Train import DataLoader, BiOptimizer, train
from Evaluation import Visualizer

device = None

model = Model(width=8, height=8).to(device)
data_loader = DataLoader(data_dir='../Data/npimage8.npy', batch_size=128, device=device)
optim = BiOptimizer(model)
visualizer = Visualizer(model, device)

optim.step()

for batch in data_loader:
    _ = model(batch[0])

train.train(model, data_loader, optim, 10)
visualizer.visualize()
visualizer.show()
