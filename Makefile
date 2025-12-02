build:
	docker build --platform linux/amd64 . -t satellite-docking

run:
	docker run --platform linux/amd64 -v $(PWD)/out:/app/out satellite-docking

clean:
	rm -rf out
