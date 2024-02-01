# RL agent to play Sonic The Hedgehog

## To add the game to gym package:
```sh
python -m retro.import games/
```

## To test sonic game:
```sh
python -m retro.examples.interactive --game SonicTheHedgehog-Genesis --state GreenHillZone.Act1 --scenario contest
```

## To test mario game:
```sh
python -m test_games.mario_play --game SuperMarioBros-v0 --state COMPLEX_MOVEMENT
```