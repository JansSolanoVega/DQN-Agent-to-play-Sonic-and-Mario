# RL agent to play Sonic The Hedgehog

## To add the game to gym package:
```sh
python -m retro.import games/
```

## To test sonic game:
```sh
python -m test_games.sonic_play --game SonicTheHedgehog-Genesis --state GreenHillZone.Act1 --scenario contest
```

## To test mario game:
```sh
python -m test_games.mario_play --game SuperMarioBros-v0 --state RIGHT_ONLY
```