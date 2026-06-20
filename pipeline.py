import yaml
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.model_step import ModelStep
from sagemaker.processing import ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline_context import PipelineSession

# 读取配置
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 初始化会话
sagemaker_session = PipelineSession(region_name=cfg["region"])
role = cfg["role_arn"]
bucket = cfg["bucket"]
prefix = cfg["prefix"]

# S3路径定义
raw_data_uri = f"s3://{bucket}/{prefix}/raw_data/"
processed_data_uri = f"s3://{bucket}/{prefix}/processed_data/"
model_output_uri = f"s3://{bucket}/{prefix}/model/"
evaluation_uri = f"s3://{bucket}/{prefix}/evaluation/"

# ---------------------- Step1 数据预处理 ----------------------
script_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve(
        framework="sklearn", region=cfg["region"], version="1.2-1"
    ),
    command=["python3"],
    instance_type=cfg["preprocess_instance_type"],
    instance_count=1,
    base_job_name="preprocess-job",
    role=role,
    sagemaker_session=sagemaker_session
)

step_preprocess = ProcessingStep(
    name="PreprocessStep",
    processor=script_processor,
    inputs=[sagemaker.processing.ProcessingInput(
        source=raw_data_uri,
        destination="/opt/ml/processing/input"
    )],
    outputs=[sagemaker.processing.ProcessingOutput(
        output_name="processed_data",
        destination=processed_data_uri,
        source="/opt/ml/processing/output"
    )],
    code="src/preprocess.py"
)

# ---------------------- Step2 模型训练 ----------------------
sklearn_estimator = SKLearn(
    entry_point="train.py",
    source_dir="src",
    role=role,
    instance_type=cfg["train_instance_type"],
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    output_path=model_output_uri,
    base_job_name="train-job",
    sagemaker_session=sagemaker_session
)

step_train = TrainingStep(
    name="TrainStep",
    estimator=sklearn_estimator,
    inputs={"train": step_preprocess.properties.ProcessingOutputs["processed_data"]}
)

# ---------------------- Step3 模型评估 ----------------------
eval_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve(
        framework="sklearn", region=cfg["region"], version="1.2-1"
    ),
    command=["python3"],
    instance_type=cfg["evaluate_instance_type"],
    instance_count=1,
    base_job_name="evaluate-job",
    role=role,
    sagemaker_session=sagemaker_session
)

step_evaluate = ProcessingStep(
    name="EvaluateStep",
    processor=eval_processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        sagemaker.processing.ProcessingInput(
            source=step_preprocess.properties.ProcessingOutputs["processed_data"],
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[sagemaker.processing.ProcessingOutput(
        output_name="evaluation",
        destination=evaluation_uri,
        source="/opt/ml/processing/evaluation"
    )],
    code="src/evaluate.py"
)

# ---------------------- Step4 条件判断：精度达标才注册模型 ----------------------
cond = ConditionGreaterThanOrEqualTo(
    left=step_evaluate.properties.ProcessingOutputs["evaluation"]["accuracy"],
    right=cfg["accuracy_threshold"]
)

# 模型注册步骤（条件满足才执行）
model_package_group_name = "DemoModelPackageGroup"
model_step = ModelStep(
    name="RegisterModel",
    model=sagemaker.model.Model(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        image_uri=sklearn_estimator.image_uri,
        role=role,
        sagemaker_session=sagemaker_session
    ),
    model_package_group_name=model_package_group_name,
    approval_status="PendingManualApproval"
)

step_condition = ConditionStep(
    name="AccuracyCheck",
    conditions=[cond],
    if_steps=[model_step],
    else_steps=[]
)

# ---------------------- 组装整条Pipeline ----------------------
pipeline = Pipeline(
    name="MLDemoPipeline",
    parameters=[],
    steps=[step_preprocess, step_train, step_evaluate, step_condition],
    sagemaker_session=sagemaker_session
)

if __name__ == "__main__":
    # 打印流水线JSON定义并上传AWS
    pipeline.upsert(role_arn=role)
    print("Pipeline 创建/更新完成，请到SageMaker控制台查看")
