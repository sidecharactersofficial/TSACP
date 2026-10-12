FROM rust:1.75 as builder
WORKDIR /usr/src/app
COPY . .
# Note: In a real environment, you'd cache dependencies better, but this works for prototype
RUN cargo build --release

FROM debian:bookworm-slim
# Install sqlite3 for potential debugging in container
RUN apt-get update && apt-get install -y sqlite3 libsqlite3-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/local/bin
COPY --from=builder /usr/src/app/target/release/fast-loop .
CMD ["./fast-loop"]
