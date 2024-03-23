sed "s/LICHESS_TOKEN/${LICHESS_TOKEN}/g" config.yml.template > config.yml
python3 lichess-bot.py
