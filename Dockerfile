FROM pytorch/pytorch
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

