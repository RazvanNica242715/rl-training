import wandb

run_id = "fhseidkq"

api = wandb.Api()
run = api.run(f'230395-breda-university-of-applied-sciences/ot2-rl-control/{run_id}')

# Download the model
run.file('model.zip').download()