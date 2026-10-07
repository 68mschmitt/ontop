.PHONY: build test test-race vet fmt fmt-check bench run clean

BINARY := ontop
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -X main.version=$(VERSION)

build:
	go build -ldflags "$(LDFLAGS)" -o $(BINARY) .

test:
	go test ./...

test-race:
	go test -race -count=1 ./...

vet:
	go vet ./...

fmt:
	gofmt -w .

fmt-check:
	@test -z "$$(gofmt -l .)" || { echo "unformatted files:"; gofmt -l .; exit 1; }

bench:
	go test -run='^$$' -bench=. ./...

run: build
	./$(BINARY)

clean:
	rm -f $(BINARY)
