import sagemaker
from sagemaker.pytorch import PyTorch

role = "arn:aws:iam::899212678931:role/service-role/AmazonSageMaker-ExecutionRole-20260405T024066"
checkpoint_s3_uri = 's3://agression-model/checkpoints/'
local_checkpoint_dir = '/opt/ml/checkpoints'

estimator = PyTorch(
    source_dir="./src",
    entry_point='train.py',
    role=role,
    #use_spot_instances=True,
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    framework_version='2.2.0',
    py_version='py310',
    distribution={
        'torch_distributed': {
            'enabled': True
        }
    },
    hyperparameters={
        'epochs': 10,
        'batch-size': 64,
        'checkpoint-dir': local_checkpoint_dir,
        'model-dir': '/opt/ml/model',
        'data-dir': '/opt/ml/data'
    },
    output_path='s3://agression-model/output/',
    checkpoint_s3_uri=checkpoint_s3_uri,
    checkpoint_local_path=local_checkpoint_dir,
)
estimator.fit({
    'training': 's3://agression-model/'
})