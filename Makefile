build:
	@docker build . -t language-models


run:
	@docker run -d -it --runtime=nvidia \
    -v ~/language-models/configs:/app/configs \
	-v ~/language-models/data:/app/data \
	-v ~/language-models/logs:/app/logs \
	-v ~/language-models/outputs:/app/outputs \
	--name language-models language-models
    useradd -u `id -u` $USER
