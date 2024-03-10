FROM python:slim
WORKDIR /app
COPY ./ /app
RUN pip3 install --no-cache-dir -r ./files/requirements.txt
CMD ["tail", "-f", "/dev/null"]
