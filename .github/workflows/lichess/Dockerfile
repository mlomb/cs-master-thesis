FROM rust:1.77 AS engine_builder

COPY engine /engine
COPY nn /nn
COPY models /models
WORKDIR /engine

ENV RUSTFLAGS="-C target-feature=+sse,+sse2,+ssse3,+sse4.1,+sse4.2,+avx,+avx2,+vnmi,+avx_vnni"
RUN cargo build --release


FROM python:3.11

RUN apt-get update && \
    apt-get install -y git python3-pip

RUN git clone https://github.com/lichess-bot-devs/lichess-bot /lichess-bot

WORKDIR /lichess-bot
RUN pip install -r requirements.txt

COPY .github/workflows/lichess/config.yml.template /lichess-bot
COPY .github/workflows/lichess/start_bot.sh /lichess-bot
COPY --from=engine_builder /engine/target/release/engine /lichess-bot

CMD ["sh", "start_bot.sh"]
