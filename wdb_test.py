import wandb
import pdb

wandb.init(project="debugging", entity="cdwer098")
wandb.define_metric("for_step", step_sync=False)
wandb.define_metric("value2", step_metric="for_step")


for i in range(0,30):
    wandb.log({"value" : 3*i}, step=i*2)
    wandb.log({"for_step" : 10*i}, step=i*2)

for i in range(5):
    wandb.log({"value2" : i ** 2})

for i in range(30,60):
    wandb.log({"value" : 3*i}, step=i*2)
    wandb.log({"for_step" : 10*i}, step=i*2)
    

pdb.set_trace()
print('end')