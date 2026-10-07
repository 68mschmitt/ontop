.PHONY: build test test-race vet fmt fmt-check bench run clean

BINARY := ontop

build:
	go build -o $(BINARY) .

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
