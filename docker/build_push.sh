cp ../dev_requirements.txt . && \
docker build --no-cache -f Dockerfile -t supervisely/embeddings-generator:0.1.5 .. && \
rm dev_requirements.txt && \
docker push supervisely/embeddings-generator:0.1.5 
