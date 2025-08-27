cp ../dev_requirements.txt . && \
docker build --no-cache -f Dockerfile -t supervisely/embeddings-generator:vis-test .. && \
rm dev_requirements.txt && \
docker push supervisely/embeddings-generator:vis-test
