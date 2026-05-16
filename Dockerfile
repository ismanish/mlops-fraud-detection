FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY configs/ ${LAMBDA_TASK_ROOT}/configs/
COPY params.yaml ${LAMBDA_TASK_ROOT}/

COPY models/ ${LAMBDA_TASK_ROOT}/models/

CMD ["src.serving.lambda_handler.handler"]
